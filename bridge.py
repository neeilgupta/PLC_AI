import asyncio
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from anthropic import AsyncAnthropic
import google.generativeai as genai
from pydantic import BaseModel
from pymodbus.client import AsyncModbusTcpClient


load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger("industrial-ai-bridge")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class TagSpec:
    register: int
    name: str
    data_type: str
    description: str
    default: int = 0


REGISTER_MAP: list[TagSpec] = [
    TagSpec(0, "Conveyor_Running", "bool", "Conveyor motor run feedback"),
    TagSpec(1, "Sensor_Blocked", "bool", "Jam/photo-eye blocked sensor"),
    TagSpec(2, "Motor_Current", "int", "Motor current in deci-amps"),
    TagSpec(3, "Safety_OK", "bool", "Safety chain healthy"),
    TagSpec(4, "Start_Command", "bool", "Start pushbutton or HMI request"),
    TagSpec(5, "Stop_Command", "bool", "Stop pushbutton or HMI request"),
    TagSpec(6, "Reset_Command", "bool", "Fault reset request"),
    TagSpec(7, "Tank_Level_Low", "bool", "Low-level switch active"),
    TagSpec(8, "Tank_Level_High", "bool", "High-level switch active"),
    TagSpec(9, "Sequence_Timeout", "bool", "Sequence or process step timeout"),
    TagSpec(10, "System_Fault_Latch", "bool", "Master latched fault"),
    TagSpec(11, "Pump_Running", "bool", "Pump run feedback"),
    TagSpec(12, "HVAC_Fault", "bool", "HVAC or remote skid fault"),
    TagSpec(13, "Mode_Code", "int", "Scenario selector / operating mode"),
    TagSpec(14, "Fault_Code", "int", "Scenario-specific fault code"),
]


MODE_LABELS = {
    0: "IDLE",
    1: "CONVEYOR",
    2: "SEQUENCE_TEST",
    3: "TANK_FILL",
    4: "HVAC_PUMP",
}

FAULT_LABELS = {
    0: "NO_FAULT",
    101: "START_BLOCKED_SAFETY",
    201: "CONVEYOR_JAM",
    301: "SEQUENCE_TIMEOUT",
    401: "TANK_FILL_VERIFY",
    501: "HVAC_OR_PUMP_FAULT",
}


@dataclass
class Settings:
    data_source: str = os.getenv("DATA_SOURCE", "modbus").strip().lower()
    modbus_host: str = os.getenv("MODBUS_HOST", "127.0.0.1")
    modbus_port: int = int(os.getenv("MODBUS_PORT", "502"))
    modbus_unit_id: int = int(os.getenv("MODBUS_UNIT_ID", "1"))
    modbus_poll_interval_s: float = float(os.getenv("MODBUS_POLL_INTERVAL_S", "0.50"))
    modbus_timeout_s: float = float(os.getenv("MODBUS_TIMEOUT_S", "1.5"))
    mock_scenario: str = os.getenv("MOCK_SCENARIO", "idle")
    fastapi_host: str = os.getenv("FASTAPI_HOST", "0.0.0.0")
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", "8000"))
    ai_model: str = os.getenv("AI_MODEL", "claude-sonnet-4-20250514")
    ai_cooldown_s: int = int(os.getenv("AI_COOLDOWN_S", "45"))
    ai_temperature: float = float(os.getenv("AI_TEMPERATURE", "0.1"))
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
    event_log_size: int = int(os.getenv("EVENT_LOG_SIZE", "250"))
    ai_history_size: int = int(os.getenv("AI_HISTORY_SIZE", "25"))
    dashboard_refresh_ms: int = int(os.getenv("DASHBOARD_REFRESH_MS", "2000"))
    cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("CORS_ORIGINS", "*").split(",")
            if origin.strip()
        ]
    )

    @property
    def ai_enabled(self) -> bool:
        return bool(self.anthropic_api_key)


@dataclass
class DiagnosticReport:
    issue_key: str
    issue_summary: str
    likely_cause: str
    severity: str
    troubleshooting_step: str
    recommended_checks: list[str]
    escalation_note: str
    classification: str
    control_vs_physical: str
    generated_by: str
    timestamp: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ChatRequest(BaseModel):
    question: str


class MockScenarioRequest(BaseModel):
    scenario: str


class WebSocketManager:
    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message)
        stale_connections: list[WebSocket] = []
        async with self._lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(payload)
                except Exception:
                    stale_connections.append(connection)
            for connection in stale_connections:
                self.active_connections.discard(connection)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)


class RuleBasedDiagnostics:
    def generate(
        self,
        issue_context: dict[str, Any],
        tags: dict[str, Any],
    ) -> DiagnosticReport:
        issue_id = issue_context["issue_id"]
        if issue_id == "motor_start_blocked":
            return DiagnosticReport(
                issue_key=issue_context["issue_key"],
                issue_summary="Motor start is blocked by the safety chain.",
                likely_cause="A start command is present, but the PLC sees Safety_OK as false so the run request is being rejected by design.",
                severity="MEDIUM",
                troubleshooting_step="Check the E-stop string, guard switches, and safety relay feedback before attempting another start.",
                recommended_checks=[
                    "Verify Safety_OK goes to 1 in the PLC watch table.",
                    "Confirm the emergency stop is released and any guard door is closed.",
                    "Check whether the start command stays on after the safety circuit returns healthy.",
                ],
                escalation_note="Escalate to controls engineering if the safety circuit is healthy in the field but the PLC still reports Safety_OK = 0.",
                classification="engineering_phase",
                control_vs_physical="This looks primarily like intended logic behavior, not an AI-controlled action. The likely root cause is either an open safety device or incorrect safety wiring/configuration.",
                generated_by="rule-fallback",
            )
        if issue_id == "sequence_timing_failure":
            return DiagnosticReport(
                issue_key=issue_context["issue_key"],
                issue_summary="A sequence step timed out before the expected sensor or level condition arrived.",
                likely_cause="The PLC timer expired while waiting for a process transition, which usually indicates an incorrect timer preset, wrong sensor mapping, or a simulated actuator that never achieved the expected state.",
                severity="MEDIUM",
                troubleshooting_step="Review the step transition condition, timer preset, and Factory I/O tag mapping for the expected sensor.",
                recommended_checks=[
                    "Confirm which sensor or level should have changed before the timeout.",
                    "Watch Sequence_Timeout, Mode_Code, and the expected sensor tag together.",
                    "Verify the timer preset matches the demo cycle time in Factory I/O.",
                ],
                escalation_note="Escalate to controls engineering if the physical sequence looks correct but the timer still expires.",
                classification="engineering_phase",
                control_vs_physical="This is usually a control-logic validation issue first. Physical causes are possible, but the timer and mapping should be checked before assuming hardware failure.",
                generated_by="rule-fallback",
            )
        if issue_id == "tank_fill_verification":
            return DiagnosticReport(
                issue_key=issue_context["issue_key"],
                issue_summary="Tank fill logic needs verification because the level state and pump state do not match the intended sequence.",
                likely_cause="The pump or tank level transitions are not matching the expected fill sequence, which may be due to incorrect sensor polarity, timer configuration, or a simulated actuator mismatch.",
                severity="LOW",
                troubleshooting_step="Validate that low and high level inputs toggle in the right order while the pump command is active.",
                recommended_checks=[
                    "Low level should request filling; high level should stop the pump.",
                    "Confirm the simulated tank in Factory I/O is mapped to the same addresses used in CODESYS.",
                    "Check that Pump_Running changes when the PLC logic commands it.",
                ],
                escalation_note="Escalate if level switches are correct but the sequence still does not settle.",
                classification="engineering_phase",
                control_vs_physical="This is mostly a process verification case. The goal is to separate bad logic assumptions from a real pump/device issue.",
                generated_by="rule-fallback",
            )
        if issue_id == "conveyor_jam_fault":
            return DiagnosticReport(
                issue_key=issue_context["issue_key"],
                issue_summary="The conveyor appears jammed and the system fault latch is active.",
                likely_cause="The blocked sensor stayed active while the conveyor was commanded to run, and motor current rose above the normal running range. That combination matches a jam or stalled product condition.",
                severity="HIGH",
                troubleshooting_step="Stop the conveyor, clear the obstruction, inspect the sensor, then reset the fault once the blocked condition is gone.",
                recommended_checks=[
                    "Verify Sensor_Blocked returns to 0 after the jam is cleared.",
                    "Compare Motor_Current against a known healthy running value.",
                    "Check for trapped product, slipping belts, or a failed photo-eye.",
                ],
                escalation_note="Escalate if the conveyor path is clear but the sensor stays blocked or current remains high.",
                classification="technician_operations",
                control_vs_physical="The PLC is reacting correctly to abnormal runtime conditions. The likely root cause is physical: jammed product, mechanical drag, or a bad sensor.",
                generated_by="rule-fallback",
            )
        if issue_id == "fault_reset_recovery":
            return DiagnosticReport(
                issue_key=issue_context["issue_key"],
                issue_summary="A latched fault is preventing restart until the fault condition is cleared and reset is performed.",
                likely_cause="The PLC fault latch is still active, so restart requests are being blocked intentionally until the root cause is removed and Reset_Command is applied.",
                severity="MEDIUM",
                troubleshooting_step="Find the active cause first, then apply reset only after the fault inputs are healthy.",
                recommended_checks=[
                    "Check Fault_Code for the original trip source.",
                    "Confirm the abnormal input is no longer active.",
                    "Press reset only after the system is back in a safe idle state.",
                ],
                escalation_note="Escalate if reset clears momentarily and the fault immediately re-latches.",
                classification="technician_operations",
                control_vs_physical="This is normal latched-fault behavior from the PLC. The underlying cause may be physical or logical depending on the active fault code.",
                generated_by="rule-fallback",
            )
        return DiagnosticReport(
            issue_key=issue_context["issue_key"],
            issue_summary="HVAC or pump-related field diagnostics issue detected.",
            likely_cause="The remote equipment fault input is active or expected pump behavior is missing, which points to a field-device issue, overload, or missing utility condition.",
            severity="HIGH" if tags.get("HVAC_Fault") else "MEDIUM",
            troubleshooting_step="Check local power, overloads, permissives, and mechanical status of the pump or HVAC equipment.",
            recommended_checks=[
                "Inspect the field skid or HVAC package for its own alarm indicator.",
                "Confirm Pump_Running feedback matches the command state.",
                "Verify utility conditions such as breaker status, airflow, or pressure switches.",
            ],
            escalation_note="Escalate to maintenance if field power and mechanical checks are normal but the remote fault remains active.",
            classification="technician_operations",
            control_vs_physical="This is most likely a real-world device issue being reported back into the control system, not a PLC execution fault.",
            generated_by="rule-fallback",
        )


class MockTagSource:
    def __init__(self, default_scenario: str) -> None:
        self._default_values = {
            spec.name: spec.default for spec in REGISTER_MAP
        }
        self._default_values["Safety_OK"] = True
        self._default_values["Mode_Label"] = "IDLE"
        self._default_values["Fault_Label"] = "NO_FAULT"
        self._scenario_started_at = time.monotonic()
        self.active_scenario = "idle"
        self.set_scenario(default_scenario)

    def list_scenarios(self) -> list[dict[str, str]]:
        return [
            {"name": "idle", "label": "Idle / healthy"},
            {"name": "traffic_phase_conflict", "label": "Traffic Signal Conflict"},
            {"name": "pump_station_failure", "label": "Water Pump Station Failure"},
            {"name": "elevator_door_fault", "label": "Elevator Door Fault"},
            {"name": "tunnel_ventilation_fault", "label": "Tunnel Ventilation Fault"},
            {"name": "garage_door_fault", "label": "Garage Door Safety Lockout"},
            {"name": "conveyor_jam", "label": "Factory Conveyor Jam"},
        ]

    def set_scenario(self, scenario: str) -> None:
        available = {item["name"] for item in self.list_scenarios()}
        if scenario not in available:
            raise ValueError(f"Unknown mock scenario: {scenario}")
        self.active_scenario = scenario
        self._scenario_started_at = time.monotonic()

    def _base(self) -> dict[str, Any]:
        values = dict(self._default_values)
        values["Mode_Code"] = 0
        values["Fault_Code"] = 0
        return values

    def get_tags(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self._scenario_started_at
        tags = self._base()
        scenario = self.active_scenario

        if scenario == "idle":
            pass
        elif scenario == "traffic_phase_conflict":
            tags.update(
                {
                    "Mode_Code": 2,
                    "Safety_OK": False,
                    "Sensor_Blocked": True,
                    "System_Fault_Latch": True,
                    "Fault_Code": 201,
                }
            )
        elif scenario == "pump_station_failure":
            tags.update(
                {
                    "Mode_Code": 4,
                    "Pump_Running": False,
                    "Tank_Level_Low": True,
                    "Tank_Level_High": False,
                    "System_Fault_Latch": True,
                    "Fault_Code": 501,
                }
            )
        elif scenario == "elevator_door_fault":
            tags.update(
                {
                    "Mode_Code": 1,
                    "Start_Command": True,
                    "Sensor_Blocked": True,
                    "Safety_OK": False,
                    "System_Fault_Latch": True,
                    "Fault_Code": 101,
                }
            )
        elif scenario == "tunnel_ventilation_fault":
            tags.update(
                {
                    "Mode_Code": 2,
                    "Sequence_Timeout": True,
                    "System_Fault_Latch": True,
                    "Safety_OK": False,
                    "Conveyor_Running": False,
                    "Fault_Code": 301,
                }
            )
        elif scenario == "garage_door_fault":
            tags.update(
                {
                    "Mode_Code": 0,
                    "Start_Command": True,
                    "Sensor_Blocked": True,
                    "Safety_OK": False,
                    "System_Fault_Latch": True,
                    "Sequence_Timeout": False,
                    "Fault_Code": 101,
                }
            )
        elif scenario == "conveyor_jam":
            tags.update(
                {
                    "Mode_Code": 1,
                    "Conveyor_Running": False,
                    "Sensor_Blocked": True,
                    "Safety_OK": False,
                    "System_Fault_Latch": True,
                    "Fault_Code": 201,
                }
            )

        tags["Mode_Label"] = MODE_LABELS.get(tags["Mode_Code"], "UNKNOWN")
        tags["Fault_Label"] = FAULT_LABELS.get(tags["Fault_Code"], "UNMAPPED")
        return tags

    def get_registers(self) -> list[int]:
        tags = self.get_tags()
        registers: list[int] = []
        for spec in REGISTER_MAP:
            value = tags[spec.name]
            if spec.data_type == "bool":
                registers.append(1 if bool(value) else 0)
            else:
                registers.append(int(value))
        return registers


class ClaudeDiagnosticsClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client: Optional[AsyncAnthropic] = None
        if settings.ai_enabled:
            self.client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self.gemini_model = None
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')

    async def create_diagnostic(
        self,
        issue_context: dict[str, Any],
        tags: dict[str, Any],
        recent_events: list[dict[str, Any]],
        current_machine_state: str,
    ) -> DiagnosticReport:
        if not self.client:
            raise RuntimeError("AI client not configured")

        system_prompt = (
            "You are an industrial diagnostics assistant. "
            "You do not control the machine. "
            "Use only the provided tags, recent events, and issue context. "
            "Be concise, practical, technician-friendly, and honest about uncertainty. "
            "Return only valid JSON with the following keys: "
            "issue_summary, likely_cause, severity, troubleshooting_step, "
            "recommended_checks, escalation_note, classification, control_vs_physical."
        )
        user_prompt = json.dumps(
            {
                "issue_context": issue_context,
                "machine_state": current_machine_state,
                "tags": tags,
                "recent_events": recent_events,
                "fault_label": FAULT_LABELS.get(tags.get("Fault_Code", 0), "UNKNOWN"),
                "mode_label": MODE_LABELS.get(tags.get("Mode_Code", 0), "UNKNOWN"),
                "requirements": {
                    "severity_values": ["LOW", "MEDIUM", "HIGH"],
                    "recommended_checks_max": 4,
                    "avoid_hallucinations": True,
                },
            },
            indent=2,
        )

        response = await self.client.messages.create(
            model=self.settings.ai_model,
            max_tokens=1024,
            temperature=self.settings.ai_temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        raw_content = response.content[0].text or "{}"
        payload = json.loads(raw_content)
        checks = payload.get("recommended_checks") or []
        if not isinstance(checks, list):
            checks = [str(checks)]
        return DiagnosticReport(
            issue_key=issue_context["issue_key"],
            issue_summary=str(payload.get("issue_summary", "No summary returned.")),
            likely_cause=str(payload.get("likely_cause", "No likely cause returned.")),
            severity=str(payload.get("severity", issue_context.get("severity", "MEDIUM"))).upper(),
            troubleshooting_step=str(payload.get("troubleshooting_step", "Review the active fault inputs and recent event transitions.")),
            recommended_checks=[str(item) for item in checks[:4]],
            escalation_note=str(payload.get("escalation_note", "Escalate if the condition cannot be reproduced safely.")),
            classification=str(payload.get("classification", issue_context.get("classification", "technician_operations"))),
            control_vs_physical=str(payload.get("control_vs_physical", "Use the tags and event history to separate logic intent from device-level behavior.")),
            generated_by="claude",
        )

    async def answer_question(
        self,
        question: str,
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.gemini_model:
            raise RuntimeError("Gemini client not configured")

        context = json.dumps({
            "machine_state": snapshot["machine"]["state_label"],
            "fault_label": snapshot["machine"]["fault_label"],
            "active_issue": snapshot.get("active_issue"),
            "tags": snapshot["tags"],
        })
        prompt = f"""You are an industrial PLC diagnostic assistant.
Answer this technician question using only the provided machine context.
Be concise, practical, and direct. Under 150 words.

Machine context: {context}

Question: {question}"""

        response = await asyncio.to_thread(
            self.gemini_model.generate_content, prompt
        )
        return {
            "answer": response.text.strip(),
            "source": "gemini",
            "timestamp": utc_now_iso(),
        }


class BridgeRuntime:
    def __init__(self, settings: Settings, ws_manager: WebSocketManager) -> None:
        self.settings = settings
        self.ws_manager = ws_manager
        self.ai_client = ClaudeDiagnosticsClient(settings)
        self.rule_diagnostics = RuleBasedDiagnostics()
        self.modbus_client: Optional[AsyncModbusTcpClient] = None
        self.mock_source: Optional[MockTagSource] = None
        self.events: deque[dict[str, Any]] = deque(maxlen=settings.event_log_size)
        self.ai_history: deque[dict[str, Any]] = deque(maxlen=settings.ai_history_size)
        self.chat_history: deque[dict[str, Any]] = deque(maxlen=40)
        self.current_tags: dict[str, Any] = {
            spec.name: spec.default for spec in REGISTER_MAP
        }
        self.current_issue: Optional[DiagnosticReport] = None
        self.issue_cache: dict[str, tuple[float, DiagnosticReport]] = {}
        self.ai_last_requested_at: dict[str, float] = {}
        self.ai_tasks: dict[str, asyncio.Task] = {}
        self.poller_task: Optional[asyncio.Task] = None
        self.last_snapshot_broadcast_at: float = 0.0
        self.connection_status = {
            "modbus": "SIMULATED" if settings.data_source == "mock" else "DISCONNECTED",
            "ai": "READY" if settings.ai_enabled else "RULE_FALLBACK",
        }
        self.last_poll_timestamp: Optional[str] = None
        self.machine_state: str = "BOOTING"
        self._last_issue_key: Optional[str] = None

    async def startup(self) -> None:
        if self.settings.data_source == "mock":
            self.mock_source = MockTagSource(self.settings.mock_scenario)
        self.add_event(
            category="system",
            message=f"Bridge runtime started using {self.settings.data_source} data source.",
            severity="INFO",
        )
        self.poller_task = asyncio.create_task(self.polling_loop(), name="modbus-poller")

    async def shutdown(self) -> None:
        for task in self.ai_tasks.values():
            task.cancel()
        if self.poller_task:
            self.poller_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.poller_task
        if self.modbus_client:
            self.modbus_client.close()

    async def polling_loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.exception("Unexpected poll loop error: %s", exc)
                self.connection_status["modbus"] = "ERROR"
                self.add_event(
                    category="modbus",
                    message=f"Polling error: {exc}",
                    severity="WARN",
                )
            await asyncio.sleep(self.settings.modbus_poll_interval_s)

    async def ensure_modbus_connection(self) -> bool:
        if self.settings.data_source == "mock":
            self.connection_status["modbus"] = "SIMULATED"
            return True

        if self.modbus_client and getattr(self.modbus_client, "connected", False):
            return True

        if self.modbus_client:
            self.modbus_client.close()

        self.modbus_client = AsyncModbusTcpClient(
            host=self.settings.modbus_host,
            port=self.settings.modbus_port,
            timeout=self.settings.modbus_timeout_s,
        )
        connected = await self.modbus_client.connect()
        self.connection_status["modbus"] = "CONNECTED" if connected else "DISCONNECTED"
        if connected:
            self.add_event(
                category="modbus",
                message=f"Connected to Modbus server at {self.settings.modbus_host}:{self.settings.modbus_port}.",
                severity="INFO",
            )
        return connected

    async def poll_once(self) -> None:
        if not await self.ensure_modbus_connection():
            return

        if self.settings.data_source == "mock":
            assert self.mock_source is not None
            registers = self.mock_source.get_registers()
        else:
            assert self.modbus_client is not None
            response = await self.modbus_client.read_holding_registers(
                address=0,
                count=len(REGISTER_MAP),
                slave=self.settings.modbus_unit_id,
            )
            if response.isError():
                self.connection_status["modbus"] = "ERROR"
                self.add_event(
                    category="modbus",
                    message=f"Modbus read error: {response}",
                    severity="WARN",
                )
                self.modbus_client.close()
                return
            registers = response.registers

        decoded = self.decode_registers(registers)
        transitions = self.detect_transitions(decoded, self.current_tags)
        self.current_tags = decoded
        self.last_poll_timestamp = utc_now_iso()
        self.machine_state = self.derive_machine_state(decoded)
        if transitions:
            for transition in transitions:
                self.add_event(
                    category="tag_transition",
                    message=transition["message"],
                    severity=transition["severity"],
                    details=transition,
                )

        issue_context = self.evaluate_issue(decoded)
        await self.update_issue_state(issue_context)
        await self.maybe_broadcast_snapshot(force=bool(transitions or issue_context))

    def decode_registers(self, registers: list[int]) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for spec in REGISTER_MAP:
            value = registers[spec.register]
            if spec.data_type == "bool":
                decoded[spec.name] = bool(value)
            else:
                decoded[spec.name] = int(value)
        decoded["Mode_Label"] = MODE_LABELS.get(decoded["Mode_Code"], "UNKNOWN")
        decoded["Fault_Label"] = FAULT_LABELS.get(decoded["Fault_Code"], "UNMAPPED")
        return decoded

    def detect_transitions(
        self,
        tags: dict[str, Any],
        previous_tags: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.last_poll_timestamp:
            return []
        transitions: list[dict[str, Any]] = []
        for key, new_value in tags.items():
            old_value = previous_tags.get(key)
            if old_value != new_value:
                severity = "INFO"
                if key in {"System_Fault_Latch", "HVAC_Fault", "Sequence_Timeout"}:
                    severity = "WARN"
                if key == "Fault_Code" and new_value not in {0, "NO_FAULT"}:
                    severity = "WARN"
                transitions.append(
                    {
                        "tag": key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "message": f"{key} changed from {old_value} to {new_value}.",
                        "severity": severity,
                        "timestamp": utc_now_iso(),
                    }
                )
        return transitions

    def derive_machine_state(self, tags: dict[str, Any]) -> str:
        if tags["System_Fault_Latch"]:
            return "FAULT_LATCHED"
        if tags["Start_Command"] and not tags["Safety_OK"]:
            return "START_BLOCKED"
        if tags["Conveyor_Running"]:
            return "RUNNING"
        if tags["Pump_Running"]:
            return "PROCESS_ACTIVE"
        return "IDLE"

    def evaluate_issue(self, tags: dict[str, Any]) -> Optional[dict[str, Any]]:
        if tags["Start_Command"] and not tags["Safety_OK"] and not tags["Conveyor_Running"]:
            return {
                "issue_id": "motor_start_blocked",
                "issue_key": f"motor_start_blocked:{int(tags['Safety_OK'])}:{int(tags['Start_Command'])}",
                "severity": "MEDIUM",
                "classification": "engineering_phase",
                "reason": "Start command present with safety chain not healthy.",
            }
        if tags["Sequence_Timeout"]:
            return {
                "issue_id": "sequence_timing_failure",
                "issue_key": f"sequence_timing_failure:{tags['Mode_Code']}:{tags['Fault_Code']}",
                "severity": "MEDIUM",
                "classification": "engineering_phase",
                "reason": "A sequence timer exceeded its expected transition window.",
            }
        if tags["System_Fault_Latch"] and tags["Sensor_Blocked"] and tags["Motor_Current"] >= 35:
            return {
                "issue_id": "conveyor_jam_fault",
                "issue_key": f"conveyor_jam_fault:{tags['Fault_Code']}:{tags['Motor_Current']}",
                "severity": "HIGH",
                "classification": "technician_operations",
                "reason": "Blocked sensor and elevated current indicate conveyor jam conditions.",
            }
        if tags["System_Fault_Latch"] and not tags["Reset_Command"]:
            return {
                "issue_id": "fault_reset_recovery",
                "issue_key": f"fault_reset_recovery:{tags['Fault_Code']}",
                "severity": "MEDIUM",
                "classification": "technician_operations",
                "reason": "Latched fault is active and reset has not been performed.",
            }
        if tags["Mode_Code"] == 3 and tags["Tank_Level_Low"] and not tags["Pump_Running"] and not tags["Tank_Level_High"]:
            return {
                "issue_id": "tank_fill_verification",
                "issue_key": f"tank_fill_verification:{int(tags['Tank_Level_Low'])}:{int(tags['Pump_Running'])}",
                "severity": "LOW",
                "classification": "engineering_phase",
                "reason": "Tank fill demand exists but pump feedback does not match the expected state.",
            }
        if tags["HVAC_Fault"] or (tags["Mode_Code"] == 4 and not tags["Pump_Running"] and tags["System_Fault_Latch"]):
            return {
                "issue_id": "hvac_pump_failure",
                "issue_key": f"hvac_pump_failure:{int(tags['HVAC_Fault'])}:{int(tags['Pump_Running'])}",
                "severity": "HIGH" if tags["HVAC_Fault"] else "MEDIUM",
                "classification": "technician_operations",
                "reason": "Remote equipment fault or missing pump behavior detected.",
            }
        return None

    async def update_issue_state(self, issue_context: Optional[dict[str, Any]]) -> None:
        if not issue_context:
            if self._last_issue_key:
                self.add_event(
                    category="diagnostics",
                    message="Active issue cleared.",
                    severity="INFO",
                )
            self.current_issue = None
            self._last_issue_key = None
            return

        issue_key = issue_context["issue_key"]
        if issue_key != self._last_issue_key:
            self.add_event(
                category="diagnostics",
                message=f"New active issue detected: {issue_context['issue_id']}.",
                severity=issue_context["severity"],
                details=issue_context,
            )
            self._last_issue_key = issue_key

        cached = self.issue_cache.get(issue_key)
        if cached:
            cached_time, cached_report = cached
            if time.monotonic() - cached_time <= self.settings.ai_cooldown_s:
                self.current_issue = cached_report
                return

        self.current_issue = self.rule_diagnostics.generate(issue_context, self.current_tags)
        if self.settings.ai_enabled:
            last_request = self.ai_last_requested_at.get(issue_key, 0.0)
            if time.monotonic() - last_request >= self.settings.ai_cooldown_s:
                self.ai_last_requested_at[issue_key] = time.monotonic()
                task = self.ai_tasks.get(issue_key)
                if not task or task.done():
                    self.ai_tasks[issue_key] = asyncio.create_task(
                        self.refresh_ai_diagnostic(issue_context),
                        name=f"ai-diagnostic-{issue_key}",
                    )

    async def refresh_ai_diagnostic(self, issue_context: dict[str, Any]) -> None:
        issue_key = issue_context["issue_key"]
        try:
            report = await self.ai_client.create_diagnostic(
                issue_context=issue_context,
                tags=self.current_tags,
                recent_events=list(self.events)[:8],
                current_machine_state=self.machine_state,
            )
            self.connection_status["ai"] = "READY"
            self.issue_cache[issue_key] = (time.monotonic(), report)
            self.current_issue = report
            self.ai_history.appendleft(report.to_dict())
            self.add_event(
                category="diagnostics",
                message=f"AI diagnostic refreshed for {issue_context['issue_id']}.",
                severity="INFO",
            )
            await self.ws_manager.broadcast(
                {
                    "type": "diagnostic",
                    "payload": report.to_dict(),
                }
            )
        except Exception as exc:
            LOGGER.warning("AI diagnostic failed, keeping rule-based result: %s", exc)
            self.connection_status["ai"] = "RULE_FALLBACK"
            self.add_event(
                category="diagnostics",
                message=f"AI fallback engaged: {exc}",
                severity="WARN",
            )

    async def maybe_broadcast_snapshot(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_snapshot_broadcast_at < 1.0:
            return
        self.last_snapshot_broadcast_at = now
        await self.ws_manager.broadcast({"type": "snapshot", "payload": self.snapshot()})

    def add_event(
        self,
        category: str,
        message: str,
        severity: str = "INFO",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.events.appendleft(
            {
                "timestamp": utc_now_iso(),
                "category": category,
                "severity": severity,
                "message": message,
                "details": details or {},
            }
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "timestamp": utc_now_iso(),
            "bridge": {
                "service": "industrial-ai-diagnostics-bridge",
                "version": "0.1.0-hackathon",
                "data_source": self.settings.data_source,
            },
            "connections": {
                **self.connection_status,
                "websocket_clients": self.ws_manager.connection_count,
                "modbus_target": (
                    "mock-scenarios"
                    if self.settings.data_source == "mock"
                    else f"{self.settings.modbus_host}:{self.settings.modbus_port}"
                ),
            },
            "machine": {
                "state_label": self.machine_state,
                "mode_code": self.current_tags.get("Mode_Code", 0),
                "mode_label": self.current_tags.get("Mode_Label", "UNKNOWN"),
                "fault_code": self.current_tags.get("Fault_Code", 0),
                "fault_label": self.current_tags.get("Fault_Label", "UNMAPPED"),
                "last_poll_timestamp": self.last_poll_timestamp,
            },
            "tags": self.current_tags,
            "active_issue": self.current_issue.to_dict() if self.current_issue else None,
            "recent_events": list(self.events)[:20],
            "ai_history": list(self.ai_history)[:10],
            "chat_history": list(self.chat_history)[:12],
            "register_map": [asdict(spec) for spec in REGISTER_MAP],
            "mock": {
                "enabled": self.settings.data_source == "mock",
                "active_scenario": self.mock_source.active_scenario if self.mock_source else None,
                "available_scenarios": self.mock_source.list_scenarios() if self.mock_source else [],
            },
        }

    async def answer_question(self, question: str) -> dict[str, Any]:
        snapshot = self.snapshot()
        self.chat_history.appendleft(
            {
                "role": "user",
                "message": question,
                "timestamp": utc_now_iso(),
            }
        )
        if self.settings.ai_enabled:
            try:
                answer = await self.ai_client.answer_question(question=question, snapshot=snapshot)
            except Exception as exc:
                LOGGER.warning("Chat AI request failed, using fallback: %s", exc)
                answer = {
                    "answer": self.fallback_chat_answer(question),
                    "source": "rule-fallback",
                    "timestamp": utc_now_iso(),
                }
        else:
            answer = {
                "answer": self.fallback_chat_answer(question),
                "source": "rule-fallback",
                "timestamp": utc_now_iso(),
            }
        self.chat_history.appendleft(
            {
                "role": "assistant",
                "message": answer["answer"],
                "timestamp": answer["timestamp"],
                "source": answer["source"],
            }
        )
        await self.ws_manager.broadcast({"type": "chat", "payload": answer})
        return answer

    def fallback_chat_answer(self, question: str) -> str:
        if self.current_issue:
            return (
                f"Current issue: {self.current_issue.issue_summary} "
                f"Likely cause: {self.current_issue.likely_cause} "
                f"Start with: {self.current_issue.troubleshooting_step}"
            )
        return (
            "No active issue is latched right now. "
            "Use the live tags and recent event list to verify whether the machine is running, idle, or waiting on a permissive."
        )


settings = Settings()
ws_manager = WebSocketManager()
runtime = BridgeRuntime(settings=settings, ws_manager=ws_manager)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.startup()
    try:
        yield
    finally:
        for task in runtime.ai_tasks.values():
            task.cancel()
        if runtime.poller_task:
            runtime.poller_task.cancel()
            with suppress(asyncio.CancelledError):
                await runtime.poller_task
        if runtime.modbus_client:
            runtime.modbus_client.close()


app = FastAPI(
    title="Industrial AI Diagnostics Bridge",
    version="0.1.0-hackathon",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.cors_origins == ["*"] else settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    snapshot = runtime.snapshot()
    return {
        "status": "ok",
        "timestamp": snapshot["timestamp"],
        "connections": snapshot["connections"],
    }


@app.get("/api/state")
async def get_state() -> dict[str, Any]:
    return runtime.snapshot()


@app.get("/api/events")
async def get_events(limit: int = Query(default=20, ge=1, le=200)) -> dict[str, Any]:
    return {
        "events": list(runtime.events)[:limit],
        "count": min(limit, len(runtime.events)),
    }


@app.get("/api/diagnostics")
async def get_diagnostics() -> dict[str, Any]:
    return {
        "active_issue": runtime.current_issue.to_dict() if runtime.current_issue else None,
        "ai_history": list(runtime.ai_history)[:10],
    }


@app.get("/api/mock/scenarios")
async def get_mock_scenarios() -> dict[str, Any]:
    if settings.data_source != "mock" or not runtime.mock_source:
        return {"enabled": False, "active_scenario": None, "scenarios": []}
    return {
        "enabled": True,
        "active_scenario": runtime.mock_source.active_scenario,
        "scenarios": runtime.mock_source.list_scenarios(),
    }


@app.post("/api/mock/scenario")
async def set_mock_scenario(request: MockScenarioRequest) -> dict[str, Any]:
    if settings.data_source != "mock" or not runtime.mock_source:
        raise HTTPException(status_code=400, detail="Mock mode is not enabled.")
    try:
        runtime.mock_source.set_scenario(request.scenario.strip())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    runtime.add_event(
        category="mock",
        message=f"Mock scenario changed to {runtime.mock_source.active_scenario}.",
        severity="INFO",
    )
    await runtime.maybe_broadcast_snapshot(force=True)
    return {
        "enabled": True,
        "active_scenario": runtime.mock_source.active_scenario,
        "snapshot": runtime.snapshot(),
    }


@app.post("/api/chat")
async def post_chat(request: ChatRequest) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    answer = await runtime.answer_question(question)
    return {
        "question": question,
        **answer,
    }


@app.websocket("/ws/live")
async def live_socket(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    await websocket.send_text(json.dumps({"type": "snapshot", "payload": runtime.snapshot()}))
    try:
        while True:
            _ = await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "bridge:app",
        host=settings.fastapi_host,
        port=settings.fastapi_port,
        reload=False,
    )
