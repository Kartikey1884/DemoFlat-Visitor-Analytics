# 🏢 FlatTrace AI — Demo Flat Visitor & Salesperson Tour Intelligence

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20OpenCV-FF4B4B.svg)](https://streamlit.io/)
[![YOLO Detection](https://img.shields.io/badge/YOLO-v26n%20%2F%20Ultralytics-00FFFF.svg)](https://docs.ultralytics.com/)
[![Tracking & ReID](https://img.shields.io/badge/Tracking-ByteTrack%20%2B%20Multi--View%20ReID-green.svg)](https://github.com/roboflow/supervision)
[![LLM Vision](https://img.shields.io/badge/LLM%20Vision-Groq%20%7C%20Gemini%20%7C%20OpenAI-orange.svg)](https://groq.com)
[![Database](https://img.shields.io/badge/Database-SQLAlchemy%20%7C%20SQLite-lightgrey.svg)](https://www.sqlalchemy.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An intelligent, real-time **Computer Vision & AI system designed for Real Estate Showrooms, Experience Centers, and Demo Flats**.

The platform automatically **traces salespersons as they conduct demo flat walkthroughs, counts and identifies visiting clients, measures tour durations, and generates structured audit logs of every client visit session**.

---

## 🎯 Purpose & Core Workflow

In real estate sales offices and experience centers, tracking client visits to model/demo flats is traditionally manual, prone to errors, and hard to audit. **FlatTrace AI** automates this end-to-end:

1. 👔 **Salesperson Recognition & Tracking**: Automatically identifies and tracks the designated salesperson via uniform/clothing signatures and LLM visual profiling.
2. 👥 **Accompanying Client Detection**: Accurately detects and counts clients touring the flat with the salesperson.
3. ⏱️ **Tour Session Logging**: Automatically creates a tour session record when a salesperson enters the flat with clients, calculating start time, end time, total dwell time, and group peak size.
4. 📋 **Audit-Ready Logs & Reports**: Stores all visit logs in the database and exports them into structured Excel sheets, CSVs, and branded PDF tour summaries.

```
                    [ Demo Flat CCTV / Live Stream / Video ]
                                       │
                                       ▼
                       [ Person Detection (YOLO26n) ]
                                       │
                                       ▼
                 [ Multi-View ReID & Clothing Color Bank ]
                                       │
                                       ▼
               [ Multimodal LLM Profiler (Groq/Gemini/OpenAI) ]
                 - Role: Sales Agent 👔 vs. Client / Visitor 👥
                 - Visual Signatures & Clothing Description
                                       │
                                       ▼
                       [ Flat Tour Session Manager ]
                 ┌───────────────────────────────────────────┐
                 │ • Traces Active Sales Agent               │
                 │ • Pairs Accompanying Client IDs           │
                 │ • Measures Tour Start, End & Duration     │
                 │ • Counts Group Size & Client Roster       │
                 └───────────────────────────────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         [ Flat Visits Log & Audit ]            [ Live Streamlit Dashboard ]
         • SQLAlchemy Database Table            • Real-time Tour Tracker HUD
         • Export: Excel / CSV / PDF            • Salesperson Performance KPIs
```

---

## ✨ Key Capabilities

### 1. 🏢 Automated Demo Flat Visit Sessions
- **Session Lifecycles**: Automatically initializes a visit session (`session_id`) when a salesperson enters the demo flat with clients.
- **Accompanying Clients Mapping**: Links unique visitor IDs (`PER-0001`, `PER-0002`, etc.) to the specific sales agent leading the tour.
- **Tour Duration & Group Sizing**: Records exact entry/exit timestamps, calculates elapsed tour duration (minutes/seconds), and detects peak group sizes.
- **Grace Period Auto-Closure**: Closes the visit session when the group departs the demo flat area after a configurable inactivity threshold.

### 2. 👔 Salesperson vs. Client Classification (ReID & LLM)
- **Multi-View ReID**: Uses multi-template memory banks (up to 12 view angles) and HSV upper/lower clothing color histograms to trace individuals seamlessly even through camera occlusions.
- **Multimodal LLM Profiler**: Integrates high-speed Vision LLMs (Groq Llama 3.3 / Llama 4 Vision, Google Gemini 1.5/2.0 Flash, OpenAI GPT-4o) to semantically classify formal attire/badges (Sales Agent) versus casual clothing (Clients).

### 3. 📊 Interactive Real Estate Analytics Dashboard
- **📋 Flat Visits Log Page**: Complete historical and in-progress log table with sales agent names, client IDs, visit duration, and session statuses.
- **🎥 Live Flat Monitor**: Real-time video feed with bounding boxes, salesperson tracking indicators, and real-time tour telemetry.
- **👥 Unique Visitors & Dwell**: Detailed client profile cards with visual attire descriptions, dwell times, and first/last seen timestamps.
- **📊 Sales Analytics**: Compare sales agent tour counts, average client tour durations, and showroom traffic across hours and days.
- **🔥 Flat Heatmaps**: Visualize hot zones within the demo flat (e.g., Living Room, Master Bedroom, Balcony) showing where clients spend the most time.

### 4. 📄 Executive & Audit Reporting
- **Excel (.xlsx)**: Formatted, multi-sheet workbook with raw visit logs, agent tour counts, and client rosters.
- **PDF Reports**: Branded executive summary with embedded charts showing tour volume and agent activity.
- **CSV Data Feeds**: Ready for direct export to real estate CRMs (Salesforce, HubSpot, LeadSquared, etc.).

---

## 📸 Dashboard Modules Overview

| Module | Purpose |
| :--- | :--- |
| 📋 **Flat Visits Log** | Real-time and historical tour log table with Sales Agent ID, Client Count, Duration, and Session IDs. |
| 🎥 **Live Monitor** | Live stream HUD highlighting Sales Agent vs. Client tracks, active tour banner, and instant snapshot captures. |
| 👥 **Unique Visitors & Dwell** | Client profiles, attire semantic tags, repeat visitor detection, and visit history. |
| 🎬 **Video Upload** | Upload and process pre-recorded demo flat CCTV footage with progress tracking. |
| 📊 **Analytics** | Aggregate tour metrics, agent tour volumes, peak hours, and client dwell distributions. |
| 🔥 **Occupancy & Heatmap** | Demo flat room utilization and decaying foot-traffic heatmaps. |
| 📄 **Reports & Export** | Generate and download CSV, styled Excel spreadsheets, and PDF tour reports. |
| ⚙️ **Settings & Zones** | Configure demo flat zones (Entrance, Living Room, Bedroom), camera settings, and LLM API keys. |

---

## 📁 Project Structure

```
FlatTrace-AI/
├── app.py                      # Main Streamlit dashboard application
├── config.py                   # Central typed configuration (settings & overrides)
├── requirements.txt            # Python dependencies
│
├── analytics/                  # Core analytics engines
│   ├── flat_visits.py          # Demo flat session manager, agent tracking & tour metrics
│   ├── customer_analytics.py   # Client dwell time and entry/exit counters
│   ├── staff_analytics.py      # Salesperson presence & active-vs-idle tracking
│   ├── table_occupancy.py      # Seating / meeting area occupancy
│   └── heatmap_generator.py    # Spatial density heatmaps
│
├── dashboard/                  # Streamlit user interface
│   ├── pages/
│   │   ├── flat_visits.py      # Dedicated Flat Visits & Sales Analytics page
│   │   ├── live.py             # Real-time camera feed & tour HUD
│   │   ├── visitors.py         # Client profiles and ReID gallery
│   │   ├── upload.py           # Video file batch processor
│   │   ├── analytics.py        # Trend charts & KPI rollups
│   │   ├── heatmap.py          # Spatial heatmaps
│   │   ├── reports.py          # Report export UI
│   │   └── settings.py         # Zone calibration & system settings
│   ├── components/             # Reusable UI cards, tables, and KPI cards
│   └── theme.py                # Dark-mode styling
│
├── database/                   # Data layer & persistence
│   ├── models.py               # SQLAlchemy schema (FlatVisitSessionModel, GlobalPersonModel, etc.)
│   └── db_manager.py           # CRUD operations and visit query handlers
│
├── detection/                  # YOLO26n object detection and annotators
├── engine/                     # Background capture pipeline & live runner
├── pipeline/                   # Orchestrator binding detection, ReID, LLM, and sessions
├── tracking/                   # ByteTrack, ReID engine, and LLM Person Profiler
├── reports/                    # Excel, PDF, and CSV report generator
└── utils/                      # Geometry, logger, and video helper utilities
```

---

## ⚡ Quickstart & Installation

### 1. Prerequisites
- **Python 3.10+** (Python 3.11 / 3.12 recommended)
- **Git**

### 2. Setup Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Install PyTorch (CPU-optimized or CUDA)
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Install dependencies
pip install -r requirements.txt
```

---

### 3. Configure LLM Vision (Optional, for Attire & Agent Profiling)

To enable automatic salesperson role identification and visual attire descriptions, set your preferred API key in a `.env` file or terminal:

```bash
# For Groq (Ultra-fast Llama 3.3 / Llama 4 Vision)
export GROQ_API_KEY="gsk_..."

# Or Google Gemini
export GEMINI_API_KEY="AIza..."

# Or OpenAI
export OPENAI_API_KEY="sk-..."
```

---

### 4. Run the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

1. Navigate to **🎥 Live Monitor** or **🎬 Video Upload**.
2. Start the feed or upload demo flat CCTV footage.
3. Switch to **📋 Flat Visits Log** to view real-time tour sessions, accompanying clients, and durations.

---

## 🧪 Verification & Testing

Verify that all session tracking, detection, and database models pass tests:

```bash
# Run full verification suite
python scripts/verify_all.py
```

---

## 🛠️ Technology Stack

- **Computer Vision & Tracking**: YOLO26n (Ultralytics), ByteTrack (Supervision), OpenCV
- **Multi-View ReID & Multimodal Vision**: HSV Clothing Signatures, Groq API (Llama 3.3/4), Google Gemini 1.5/2.0 Flash, OpenAI GPT-4o
- **Application & Dashboard**: Streamlit, Plotly, Pandas
- **Persistence & Reports**: SQLite, SQLAlchemy 2.0, openpyxl, fpdf2

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.
#   D e m o F l a t - V i s i t o r - A n a l y t i c s  
 