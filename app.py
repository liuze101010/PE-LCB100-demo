from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .mcp_bridge import DemoService


PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_SNAPSHOT = PACKAGE_ROOT / "data" / "demo_snapshot.json"
STATIC_ROOT = PACKAGE_ROOT / "static"


def _load_dotenv(path: Path) -> None:
    """Minimal, dependency-free .env loader; never overrides existing env vars."""
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


class DryRunRequest(BaseModel):
    execute: bool = False


class RouteSelectionRequest(BaseModel):
    repair_tool: str
    parameters: dict[str, object] | None = None
    rationale: str
    execute: bool = False


class AutoRunRequest(BaseModel):
    goal: str = ""


class AttributionRequest(BaseModel):
    case_id: str


class AutoSmokeRequest(BaseModel):
    case_id: str
    mode: str = "reproduce"
    confirm: bool = False
    traj: str | None = None
    bridge_dir: str | None = None


def _detect_project_root() -> Path | None:
    configured = os.getenv("POLYMER_AGENT_ROOT")
    if configured:
        return Path(configured)
    current = PACKAGE_ROOT
    for parent in (current, *current.parents):
        if (parent / "generated_code" / "md_analysis").exists():
            return parent
    return None


def create_app(
    snapshot_path: Path | str = DEFAULT_SNAPSHOT,
    project_root: Path | str | None = None,
    force_mode: str | None = None,
    *,
    mode: str | None = None,
) -> FastAPI:
    """Create an offline demo, or the explicitly selected research platform.

    ``mode`` selects the application. ``force_mode`` keeps the existing
    replay/connected MCP selection contract for the platform.
    """
    application_mode = (mode or os.getenv("POLYMER_AGENT_APP_MODE") or "demo").strip().lower()
    if application_mode not in {"demo", "platform"}:
        raise ValueError("POLYMER_AGENT_APP_MODE must be demo or platform")
    if application_mode == "platform":
        _load_dotenv(PACKAGE_ROOT / ".env")
        service_root = project_root or _detect_project_root()
        service_mode = force_mode or os.getenv("POLYMER_AGENT_DEMO_MODE")
    else:
        if force_mode not in {None, "replay"}:
            raise ValueError("Connected MCP access requires mode='platform'")
        service_root = None
        service_mode = "replay"
    service = DemoService(
        snapshot_path=snapshot_path,
        project_root=service_root,
        force_mode=service_mode,
        allow_live=application_mode == "platform",
    )
    app = FastAPI(
        title="面向聚合物分子动力学模拟的智能体框架",
        version="1.2.2",
        docs_url="/api/docs",
        redoc_url=None,
    )
    app.state.application_mode = application_mode
    app.state.demo_service = service
    if application_mode == "platform":
        from .control_plane import init_database
        from .control_plane.api import router as control_plane_router
        from .control_plane.config import load_settings
        from .control_plane.db import SessionLocal
        from .control_plane.services import bootstrap_admin

        init_database()
        with SessionLocal() as db:
            bootstrap_admin(db, load_settings())
        app.include_router(control_plane_router)

    @app.get("/api/health")
    def health():
        return {**service.health(), "application_mode": application_mode}

    @app.get("/api/overview")
    def overview():
        return service.overview()

    @app.get("/api/cases")
    def cases():
        return service.list_cases()

    @app.get("/api/audit-ledger")
    def audit_ledger():
        return service.audit_ledger()

    @app.get("/api/cases/{case_id}")
    def case_detail(case_id: str):
        try:
            return service.get_case(case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case_not_found") from exc

    @app.post("/api/cases/{case_id}/dry-run")
    def dry_run(case_id: str, request: DryRunRequest):
        try:
            return service.dry_run(case_id, execute=request.execute)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case_not_found") from exc
        except PermissionError as exc:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "mutating_request_rejected",
                    "message": str(exc),
                },
            ) from exc

    @app.get("/api/cases/{case_id}/route-options")
    def route_options(case_id: str):
        try:
            return service.route_options(case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case_not_found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/cases/{case_id}/route-selection")
    def route_selection(case_id: str, request: RouteSelectionRequest):
        try:
            return service.select_route(
                case_id,
                repair_tool=request.repair_tool,
                parameters=request.parameters,
                rationale=request.rationale,
                execute=request.execute,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case_not_found") from exc
        except PermissionError as exc:
            raise HTTPException(
                status_code=403,
                detail={"code": "mutating_request_rejected", "message": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "route_selection_rejected", "message": str(exc)},
            ) from exc

    @app.post("/api/auto/run")
    def auto_run(request: AutoRunRequest):
        return service.auto_run(request.goal)

    @app.get("/api/auto/report")
    def auto_report(format: str = "md", goal: str = ""):
        try:
            artifact = service.auto_report(goal, format)
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "unsupported_format", "message": str(exc)},
            ) from exc
        return Response(
            content=artifact["content"],
            media_type=artifact["media_type"],
            headers={
                "Content-Disposition": f'attachment; filename="{artifact["filename"]}"'
            },
        )

    @app.post("/api/auto/attribution")
    def auto_attribution(request: AttributionRequest):
        try:
            return service.run_attribution(request.case_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case_not_found") from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=502,
                detail={"code": "attribution_failed", "message": str(exc)},
            ) from exc

    @app.post("/api/auto/smoke")
    def auto_smoke(request: AutoSmokeRequest):
        try:
            return service.start_smoke(
                request.case_id,
                mode=request.mode,
                confirm=request.confirm,
                traj=request.traj,
                bridge_dir=request.bridge_dir,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="case_not_found") from exc
        except PermissionError as exc:
            raise HTTPException(
                status_code=403,
                detail={"code": "confirm_required", "message": str(exc)},
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"code": "bad_request", "message": str(exc)},
            ) from exc

    @app.get("/api/auto/smoke/{job_id}")
    def auto_smoke_status(job_id: str):
        try:
            return service.smoke_status(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job_not_found") from exc

    if STATIC_ROOT.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_ROOT), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_ROOT / ("platform.html" if application_mode == "platform" else "auto.html"))

    @app.get("/demo", include_in_schema=False)
    def demo_page():
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/auto", include_in_schema=False)
    def auto_page():
        return FileResponse(STATIC_ROOT / "auto.html")

    return app


if DEFAULT_SNAPSHOT.exists():
    app = create_app()
else:
    app = FastAPI(title="面向聚合物分子动力学模拟的智能体框架：本地数据待准备")
