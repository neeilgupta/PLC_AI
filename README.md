#Industrial Fault Diagnostic System


PLC.AI is a real-time fault diagnostic tool for industrial Programmable Logic Controllers. When a PLC faults on a factory floor, technicians waste hours manually reading fault codes and digging through documentation. PLC.AI detects the fault in real time, surfaces it on a dashboard, and uses Claude AI to stream back structured troubleshooting guidance in under 3 seconds.

---

## How It Works

```
CodeSys PLC (teammate laptop)
    │  Modbus TCP (port 502)
    ▼
ESP32 (Espressif hardware)
    │  edge fault classifier runs locally
    │  WebSocket /ws/live
    ▼
bridge.py (Neeil's laptop — FastAPI)
    │  polls Modbus every 0.5s
    │  calls Claude API for structured diagnostics
    │  serves REST + WebSocket
    ▼
frontend/index.html (browser dashboard)
    │  polls /api/state every 1s
    │  displays live tag monitor + AI diagnostic
```

All devices connect through **Neeil's phone hotspot** — do not use venue Wi-Fi (client isolation blocks device-to-device traffic).

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

### Environment

Copy `.env.example` to `.env` and fill in your keys:

```env
DATA_SOURCE=mock           # use "mock" until ESP32 is confirmed working
MOCK_SCENARIO=idle

ANTHROPIC_API_KEY=sk-ant-...   # Claude — fault diagnostics
AI_MODEL=claude-sonnet-4-20250514

MODBUS_HOST=192.168.1.x    # teammate's CodeSys laptop IP
MODBUS_PORT=502
```

### Run

```bash
python bridge.py
```

Then open `frontend/index.html` in a browser, or navigate to `http://localhost:8000`.

---

## Demo Scenarios

The dashboard has buttons to switch between 6 mock scenarios — no hardware needed.

| Button | Fault Code | Story |
|--------|-----------|-------|
| TRAFFIC CONFLICT | 201 | N/S and E/W signals go green simultaneously — conflict monitor trips |
| PUMP FAILURE | 501 | Primary pump trips at 2am, backup fails to auto-start, tank level dropping |
| ELEVATOR DOOR | 101 | Door close commanded, sensor never clears, car stuck at floor 4 |
| TUNNEL VENT | 301 | CO2 above threshold, fan command sent, no run feedback — sequence timeout |
| GARAGE DOOR | 101 | Obstruction sensor trips mid-cycle, door reverses twice, safety lockout |
| CONVEYOR JAM | 201 | Belt stopped while motor running, jam trips safety interlock |

Click **IDLE** to clear the active fault and reset the dashboard.

---

## File Structure

```
PLC_AI/
├── bridge.py              # FastAPI backend — Modbus polling, Claude diagnostics, WebSocket
├── requirements.txt       # Python dependencies
├── .env                   # API keys and config (not committed)
├── .env.example           # Template
├── frontend/
│   └── index.html         # Single-file dashboard — no build step, no framework
└── claude_docs/
    ├── masterdoc.md       # Full project context and architecture
    └── BUILD_PROGRESS.md  # Session-by-session build log
```

> `plcai/` — original reference files from the hackathon starter kit. Do not edit.  
> `StarkHacks-Guide-for-PLC-Troubleshooting-Tool/` — reference only.

---

## Network Setup (Day Of)

1. All devices connect to **Neeil's phone hotspot**
2. ESP32 static IP: `192.168.1.100` (hardcoded in firmware)
3. Check Neeil's laptop IP: `ifconfig | grep "inet "`
4. Update `MODBUS_HOST` in `.env` to the CodeSys laptop's IP

Connection test:
```bash
ping 192.168.1.100                      # ESP32 reachable?
curl http://localhost:8000/health       # bridge.py running?
curl http://localhost:8000/api/state    # data flowing?
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Bridge status and connection info |
| GET | `/api/state` | Full system snapshot — tags, machine state, active fault, AI diagnostic |
| GET | `/api/events` | Recent event log (last 250) |
| GET | `/api/diagnostics` | Active issue + AI history |
| POST | `/api/mock/scenario` | Switch mock scenario `{"scenario": "conveyor_jam"}` |
| WS | `/ws/live` | WebSocket — ESP32 connects here |

---

## Register Map

All 15 Modbus holding registers read by the ESP32 and exposed via `/api/state`:

| Register | Tag | Type |
|----------|-----|------|
| HR0 | `Conveyor_Running` | bool |
| HR1 | `Sensor_Blocked` | bool |
| HR2 | `Motor_Current` | int (deci-amps) |
| HR3 | `Safety_OK` | bool |
| HR4 | `Start_Command` | bool |
| HR5 | `Stop_Command` | bool |
| HR6 | `Reset_Command` | bool |
| HR7 | `Tank_Level_Low` | bool |
| HR8 | `Tank_Level_High` | bool |
| HR9 | `Sequence_Timeout` | bool |
| HR10 | `System_Fault_Latch` | bool |
| HR11 | `Pump_Running` | bool |
| HR12 | `HVAC_Fault` | bool |
| HR13 | `Mode_Code` | int |
| HR14 | `Fault_Code` | int |

---

## Fallback Ladder

If something breaks during the demo, drop to the next level:

| Level | State | What still works |
|-------|-------|-----------------|
| 1 | Full pipeline | Auto-detect from CodeSys, real Claude response |
| 2 | ESP32/CodeSys down | `DATA_SOURCE=mock` in `.env`, mock buttons work |
| 3 | bridge.py down | Manual scenario buttons in frontend, cached responses fire |
| 4 | Everything down | Open `frontend/index.html` directly in browser — no server needed |

**Level 4 always works. Never have nothing to show.**

---


