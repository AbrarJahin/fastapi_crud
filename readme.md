# FastAPI SQLite CRUD Project

A clean, modular **FastAPI** CRUD application using:

- **Python 3.9.12**
- **FastAPI**
- **SQLAlchemy**
- **SQLite**
- **Alembic** for migrations
- **Makefile** for workflow automation
- **YAML-based logging** with per-run log files

This project is structured for learning, coursework, and real-world best practices.

---

## 📁 Project Structure

```
.
├── app/
│   ├── api/
│   │   └── items.py              # Item CRUD APIs
│   ├── models/
│   │   ├── db/
│   │   │   └── item.py           # SQLAlchemy DB model
│   │   └── view/
│   │       └── items.py          # Pydantic view models
│   ├── database.py               # DB engine & session
│   └── main.py                   # App entry point
│
├── alembic/
│   ├── versions/                 # Migration files
│   └── env.py                    # Alembic config
│
├── logging.yaml                  # Base logging config
├── Makefile                      # Dev / DB / migration commands
├── requirements.txt
├── alembic.ini
├── app.sqlite3                   # SQLite DB (runtime)
├── .log/                         # Auto-created log folder
└── README.md
```

---

## 🚀 Features

- Full CRUD APIs for `Item`
- Pagination support
- Proper separation of concerns:
  - API layer
  - DB models
  - View (schema) models
- Alembic-based schema migrations
- One log file per app run
- Swagger & ReDoc auto-generated docs
- Makefile-driven workflow

---

## 🧪 API Endpoints

| Method | Endpoint            | Description             |
|------:|---------------------|-------------------------|
| GET   | `/health`           | Health check            |
| POST  | `/items`            | Create item             |
| GET   | `/items/{id}`       | Get item by ID          |
| GET   | `/items`            | List items (pagination) |
| PUT   | `/items/{id}`       | Update item             |
| DELETE| `/items/{id}`       | Delete item             |

---

## 📘 API Documentation

Once running:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 🛠️ Setup Instructions

### 1️⃣ Prerequisites

Before running this project, ensure the following tools are installed on your system:

#### Required
- **Python 3.9.12**
  - Verify:
    ```bash
    python3.9 --version
    ```
- **pip** (comes with Python)
  - Verify:
    ```bash
    pip --version
    ```
- **Make**
  - macOS / Linux:
    ```bash
    make --version
    ```
  - Windows:
    - Install via **Chocolatey**:
      ```powershell
      choco install make
      ```
    - Or use **Git Bash** (recommended)

#### Recommended
- **VS Code**
- **VS Code Python extension**
- **SQLite browser** (for inspecting `app.sqlite3`)
- **curl** or **Postman** (API testing)
---

### 2️⃣ Install Dependencies

```bash
make install
```

---

## ▶️ Running the Application

### Development Mode

```bash
make dev
```

This command:
- Creates `.log/` if missing
- Generates runtime logging config
- Starts FastAPI with auto-reload

---

## 🧾 Logging

- Logging is configured using `logging.yaml`
- One log file is generated per run
- Log filename format:

```
.log/YYYY_MM_DD_HH_MM_SS-PID.log
```

Example:

```
.log/2025_12_29_21_43_07-18240.log
```

---

## 🗄️ Database & Migrations (Alembic)

### Initialize Database (first time)

```bash
make init-db
```

### Create a new migration

```bash
make migrate message="add_new_table"
```

### Apply migrations

```bash
make update-db
```

---

## 📦 Makefile Commands

| Command | Description |
|-------|-------------|
| `make dev` | Run FastAPI app with logging |
| `make install` | Install dependencies |
| `make init-db` | Initialize Alembic |
| `make migrate message="msg"` | Create migration |
| `make update-db` | Apply migrations |
| `make lint` | Run linting |
| `make format` | Format code |

---

## 🧹 VS Code Workspace Cleanup (Optional)

Hidden via workspace-only settings:
- `.vscode/`
- `.conda/`
- `app.sqlite3`

Configured in:

```
.vscode/settings.json
```

---

## 📌 Notes & Best Practices

- SQLite is used for simplicity
- Easy migration to PostgreSQL later
- Clear separation of DB models and API schemas
- Router-based API design
- PID-based logging for debugging

---

## 🧩 Possible Improvements

- Add service layer
- Add unit tests (`pytest`)
- Dockerize the application
- Add authentication
- Switch to async SQLAlchemy

---

## 👨‍💻 Author

Built as a clean FastAPI reference project for learning.
