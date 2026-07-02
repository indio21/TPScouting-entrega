"""Rutas operativas de configuracion."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import List, Optional

from flask import Blueprint, current_app, flash, render_template, request


def create_settings_blueprint(*, deps: SimpleNamespace) -> Blueprint:
    bp = Blueprint("settings", __name__)

    @bp.route("/settings", methods=["GET", "POST"])
    @deps.roles_required(deps.ROLE_ADMIN)
    def settings():
        if request.method == "POST":
            deps.require_csrf()

        status_messages: List[str] = []
        modal_message: Optional[str] = None

        if request.method == "POST":
            action = request.form.get("action")
            if action == "update_database":
                start = time.time()
                if not deps.pipeline_lock.acquire(blocking=False):
                    status_messages = ["Ya hay una actualizacion en curso. Reintenta en unos minutos."]
                    flash("Ya hay una actualizacion en curso. Reintenta en unos minutos.", "warning")
                else:
                    success = False
                    try:
                        success, logs = deps.update_database_pipeline(
                            limit=deps.EVAL_POOL_MAX,
                            sync_shortlist=deps.SYNC_SHORTLIST_ENABLED,
                        )
                        status_messages = logs
                    except Exception as exc:
                        current_app.logger.exception("Pipeline update_database failed unexpectedly")
                        status_messages = [f"No se pudo completar la actualizacion: {exc}"]
                    finally:
                        duration = round(time.time() - start, 2)
                        status_messages.append(f"Duracion total de la actualizacion: {duration}s")
                        try:
                            current_app.logger.info("Pipeline update_database finished in %ss", duration)
                        except Exception:
                            pass
                        deps.pipeline_lock.release()

                    if success:
                        if deps.SYNC_SHORTLIST_ENABLED:
                            modal_message = "Se actualizaron puntajes y se sincronizo la base operativa."
                        else:
                            modal_message = "Se actualizaron puntajes (sin sincronizar jugadores operativos)."
                        flash("Listo: la actualizacion general finalizo correctamente.", "success")
                    else:
                        flash("No se pudo completar la actualizacion. Revisa el detalle.", "danger")
            elif action == "cleanup_demo_data":
                db = deps.Session()
                try:
                    summary = deps.cleanup_operational_data(db)
                    db.commit()
                    deps.invalidate_dashboard_cache()
                    status_messages = [
                        "Auditoria y limpieza de base operativa completadas.",
                        f"Jugadores invalidos removidos: {summary['removed_invalid_players']}",
                        f"Fotos completadas: {summary['photo_updates']}",
                        f"Historial tecnico sincronizado: {summary['history_updates']}",
                        f"Jugadores recortados por limite operativo: {summary['trimmed_players']}",
                        (
                            "Calidad actual -> "
                            f"total={summary['players_total']}, "
                            f"sin_dni={summary['missing_national_id']}, "
                            f"sin_nacimiento={summary['missing_birth_date']}, "
                            f"edad_invalida={summary['invalid_age']}, "
                            f"sin_nombre={summary['missing_name']}, "
                            f"sin_posicion={summary['missing_position']}, "
                            f"sin_foto={summary['missing_photo_url']}, "
                            f"stats_huerfanos={summary['orphan_stats']}, "
                            f"historial_huerfano={summary['orphan_attribute_history']}, "
                            f"exceso_limite={summary['over_limit_players']}"
                        ),
                    ]
                    if summary["removed_invalid_players"] or summary["trimmed_players"] or summary["photo_updates"]:
                        flash("Listo: la base operativa quedo auditada y consistente para la demo.", "success")
                    else:
                        flash("No se detectaron inconsistencias nuevas en la base operativa.", "info")
                except Exception as exc:
                    db.rollback()
                    status_messages = [f"No se pudo completar la limpieza operativa: {exc}"]
                    flash("No se pudo completar la limpieza de la base operativa.", "danger")
                finally:
                    db.close()

        db = deps.Session()
        try:
            data_quality = deps.compute_operational_data_quality(db)
        finally:
            db.close()
        return render_template(
            "settings.html",
            status_messages=status_messages,
            modal_message=modal_message,
            data_quality=data_quality,
            eval_pool_max=deps.EVAL_POOL_MAX,
        )

    return bp
