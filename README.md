# TMM Meter Preprocess Service

Сервіс препроцесингу фото електролічильників для покращення точності OCR.
Приймає фото (base64), збільшує + підвищує контраст + різкість, повертає покращене (base64).

## Ендпоінти

- `GET /` — перевірка живості (health check).
- `POST /preprocess` — обробка фото.

**Запит** (`POST /preprocess`):
```json
{
  "image_base64": "<фото у base64 без префікса data:>",
  "scale": 2.0,
  "clip_limit": 2.5,
  "sharpen": true
}
```

**Відповідь**:
```json
{
  "image_base64": "<покращене фото у base64>",
  "width": 1440,
  "height": 2560
}
```

## Деплой на Railway (покроково)

### Спосіб 1 — через GitHub (рекомендований)
1. Створи новий репозиторій на GitHub (напр. `tmm-meter-preprocess`).
2. Заванаж туди всі файли цієї теки (main.py, requirements.txt, railway.json, nixpacks.toml, README.md).
3. На railway.app → **New Project** → **Deploy from GitHub repo** → вибери цей репозиторій.
4. Railway автоматично збудує (Nixpacks підхопить nixpacks.toml із системними бібліотеками для OpenCV).
5. Після деплою → вкладка **Settings** → **Networking** → **Generate Domain**. Отримаєш публічний URL виду `https://tmm-meter-preprocess-production.up.railway.app`.

### Спосіб 2 — через Railway CLI
1. Встанови CLI: `npm i -g @railway/cli`
2. `railway login`
3. У цій теці: `railway init` → `railway up`
4. Згенеруй домен у Settings → Networking.

## Перевірка після деплою
Відкрий у браузері `https://<твій-домен>/` — має показати `{"status":"ok","service":"meter-preprocess"}`.

## Інтеграція в n8n
Між вузлами **To Base64** і **OCR Claude** додається HTTP-вузол, що викликає `POST /preprocess`
з `image_base64` від To Base64, а OCR Claude бере покращений `image_base64` з відповіді сервіса.

## Технічні нотатки
- `opencv-python-headless` — версія без GUI (для сервера).
- `nixpacks.toml` ставить `libgl1` + `libglib2.0-0` — без них cv2 впаде на імпорті.
- Рецепт обробки (2x + CLAHE + різкість) підібраний на реальних фото лічильників ТММ.
- Параметри scale/clip_limit можна тюнити через тіло запиту без редеплою.
