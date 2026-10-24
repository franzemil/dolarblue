"""Rotación de la DBB_API_KEY contra api.dolarbluebolivia.click.

Login con tu cuenta (getpass — la contraseña nunca pasa por el chat ni se
guarda) -> listado de API keys -> renew de la clave elegida -> la NUEVA clave
se escribe directamente en .env (solo se muestra su último cuarto).

Uso:
    uv run python -m collectors.rotate_key
"""

from __future__ import annotations

import getpass
import re
import sys
from pathlib import Path

import requests

BASE = "https://api.dolarbluebolivia.click"
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
SENSITIVE_HINTS = ("key", "token", "secret")


def mask(value: str) -> str:
    value = str(value)
    return value[:6] + "…" + value[-4:] if len(value) > 12 else "…"


def scrub(obj):
    """Enmascara valores de campos sensibles para poder imprimir con seguridad."""
    if isinstance(obj, dict):
        return {
            k: (mask(v) if isinstance(v, str) and any(h in k.lower() for h in SENSITIVE_HINTS) else scrub(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    return obj


def find_token(payload) -> str | None:
    if isinstance(payload, dict):
        for k, v in payload.items():
            if isinstance(v, str) and ("token" in k.lower() or "bearer" in k.lower()):
                if len(v) > 20:
                    return v
        for v in payload.values():
            if (found := find_token(v)) and found:
                return found
    return None


def main() -> None:
    email = input("Email de tu cuenta DBB: ").strip()
    password = getpass.getpass("Contraseña (no se muestra): ")

    s = requests.Session()
    s.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    r = s.post(f"{BASE}/private/v1/auth/login", json={"email": email, "password": password}, timeout=30)
    if not r.ok:
        sys.exit(f"login falló ({r.status_code}): {scrub(r.json() if r.ok else r.text[:200])}")
    bearer = find_token(r.json())
    if not bearer:
        sys.exit(f"No encontré el token de sesión en la respuesta: {scrub(r.json())}")
    s.headers["Authorization"] = f"Bearer {bearer}"
    print("login OK ✓")

    r = s.get(f"{BASE}/private/v1/api-keys", timeout=30)
    if not r.ok:
        sys.exit(f"listado falló ({r.status_code})")
    keys = r.json()
    rows = keys if isinstance(keys, list) else keys.get("items", keys.get("api_keys", keys.get("keys", [])))
    print("\nTus API keys (valores enmascarados):")
    print(scrub(rows))

    key_id = input("\nkey_id a renovar (o Enter para crear una nueva): ").strip()
    if key_id:
        r = s.post(f"{BASE}/private/v1/api-keys/{key_id}/renew", timeout=30)
    else:
        r = s.post(
            f"{BASE}/private/v1/api-keys",
            json={"label": "maestria-ds", "weekly_allotment": None},
            timeout=30,
        )
    if not r.ok:
        sys.exit(f"operación falló ({r.status_code}): {r.text[:300]}")

    new_key = find_token(r.json())
    if not new_key:
        sys.exit(f"La respuesta no trae la clave nueva: {scrub(r.json())}")
    print(f"\nclave nueva generada ✓ ({mask(new_key)})")

    env_text = ENV_PATH.read_text(encoding="utf-8")
    if "DBB_API_KEY" in env_text:
        env_text = re.sub(r"DBB_API_KEY=.*", f"DBB_API_KEY={new_key}", env_text)
    else:
        env_text += f"\nDBB_API_KEY={new_key}\n"
    ENV_PATH.write_text(env_text, encoding="utf-8")
    print(f".env actualizado ✓ ({ENV_PATH})")

    test = requests.get(
        f"{BASE}/private/v1/rates/official", headers={"X-API-Key": new_key}, timeout=30
    )
    print(f"test con la clave nueva: {'OK ✓' if test.ok else f'FALLÓ ({test.status_code})'}")

    if key_id:
        print("\nOpcional: revoca las sesiones activas viejas con POST /private/v1/auth/sessions/revoke-others")


if __name__ == "__main__":
    main()
