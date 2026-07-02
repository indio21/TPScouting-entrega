import os
import re
import threading
from datetime import date

import pytest
from werkzeug.serving import make_server

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PLAYWRIGHT") != "1",
    reason="Playwright visual smoke is opt-in. Set RUN_PLAYWRIGHT=1 to run it.",
)

sync_api = pytest.importorskip("playwright.sync_api")
expect = sync_api.expect


@pytest.fixture()
def live_server(app_module):
    server = make_server("127.0.0.1", 0, app_module.app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _create_visual_user_and_player(app_module) -> None:
    session = app_module.Session()
    try:
        user = app_module.User(
            username="visual_admin",
            password_hash=app_module.generate_password_hash("visual1234"),
            role=app_module.ROLE_ADMIN,
        )
        player = app_module.Player(
            name="Visual Juvenil",
            national_id="88990011",
            age=16,
            birth_date=date(2010, 3, 21),
            position="Mediocampista",
            club="Club Visual",
            country="Argentina",
            photo_url="",
            pace=12,
            shooting=10,
            passing=14,
            dribbling=13,
            defending=9,
            physical=12,
            vision=15,
            tackling=8,
            determination=16,
            technique=14,
            potential_label=True,
        )
        session.add_all([user, player])
        session.commit()
    finally:
        session.close()


def _has_no_horizontal_overflow(page) -> bool:
    return bool(
        page.evaluate(
            "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 2"
        )
    )


def test_login_and_dashboard_render_in_real_browser(app_module, live_server, tmp_path):
    _create_visual_user_and_player(app_module)

    with sync_api.sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            desktop = browser.new_context(viewport={"width": 1366, "height": 768})
            page = desktop.new_page()
            page.goto(f"{live_server}/login", wait_until="networkidle")
            expect(page.get_by_role("heading", name="TPScouting")).to_be_visible()
            assert _has_no_horizontal_overflow(page)

            page.fill("#username", "visual_admin")
            page.fill("#password", "visual1234")
            page.get_by_role("button", name=re.compile("Entrar")).click()
            page.wait_for_url(re.compile(r".*/players.*"))

            page.goto(f"{live_server}/dashboard", wait_until="networkidle")
            expect(page.get_by_role("heading", name="Mesa de scouting")).to_be_visible()
            assert _has_no_horizontal_overflow(page)
            dashboard_shot = tmp_path / "dashboard-desktop.png"
            page.screenshot(path=str(dashboard_shot), full_page=True)
            assert dashboard_shot.stat().st_size > 1000
            desktop.close()

            mobile = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True)
            mobile_page = mobile.new_page()
            mobile_page.goto(f"{live_server}/login", wait_until="networkidle")
            expect(mobile_page.get_by_role("heading", name="TPScouting")).to_be_visible()
            expect(mobile_page.get_by_label("Nombre de usuario")).to_be_visible()
            assert _has_no_horizontal_overflow(mobile_page)
            login_shot = tmp_path / "login-mobile.png"
            mobile_page.screenshot(path=str(login_shot), full_page=True)
            assert login_shot.stat().st_size > 1000
            mobile.close()
        finally:
            browser.close()
