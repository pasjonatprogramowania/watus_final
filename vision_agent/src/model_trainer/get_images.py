import os
import math
import time
import random
import shutil
import mimetypes
from pathlib import Path
from io import BytesIO

import requests
from PIL import Image
from tqdm import tqdm
from ddgs import DDGS

# Spróbuj zaimportować dokładną klasę wyjątku; jeśli brak – fallback do Exception
try:
    from ddgs.exceptions import RateLimitException as DDGRateLimit
except Exception:
    class DDGRateLimit(Exception):
        pass

# -------------------------
# Konfiguracja
# -------------------------
OBJECTS = ["personal computer"]
MAX_PER_CLASS = 1000

RAW_DIR = Path("raw_images")                # tymczasowy katalog na pobrane obrazy (zorganizowane wg klas)
DATASET_DIR = Path("dataset")               # wynikowy katalog z podziałem train/val/test
SPLITS = ("train", "val", "test")
TRAIN_PCT = 0.90                            # 90% train

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
TIMEOUT = 20

# Tempo – spokojniejsze wartości zmniejszają ryzyko RateLimit
REQUESTS_PER_SECOND = 1.2                   # dla pobrań plików (do serwerów z obrazami)
SLEEP_BETWEEN = 1.0 / REQUESTS_PER_SECOND
DDG_BASE_SLEEP = 2.0                        # pauza bazowa między kolejnymi zapytaniami do DDG (sek)
DDG_MAX_BACKOFF = 60.0                      # maksymalna pauza backoff (sek)

# Obsługiwane rozszerzenia obrazów (fallback do .jpg)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# -------------------------
# Pomocnicze
# -------------------------
def safe_filename(name: str) -> str:
    keep = "-_.() "
    out = "".join(ch if ch.isalnum() or ch in keep else "_" for ch in name).strip()
    out = "_".join(out.split())
    return out

def guess_extension_from_url_or_type(url: str, content_type: str | None) -> str:
    ext = Path(url.split("?")[0]).suffix.lower()
    if ext in ALLOWED_EXTS:
        return ext
    if content_type:
        ext2 = mimetypes.guess_extension(content_type.split(";")[0].strip())
        if ext2 in ALLOWED_EXTS:
            return ext2
    return ".jpg"

def is_image_content(content_type: str | None) -> bool:
    return bool(content_type and content_type.lower().startswith("image/"))

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def validate_image_bytes(b: bytes) -> bool:
    try:
        with Image.open(BytesIO(b)) as im:
            im.verify()
        return True
    except Exception:
        return False

DDG_IMPL = None
try:
    # Nowa biblioteka (pip install ddgs)
    from ddgs import DDGS as _DDGS
    DDG_IMPL = "ddgs"
except Exception:
    try:
        # Starsza/nowsza gałąź (pip install duckduckgo_search)
        from duckduckgo_search import DDGS as _DDGS
        DDG_IMPL = "duckduckgo_search"
    except Exception as e:
        raise RuntimeError("Brak zainstalowanej biblioteki ddgs/duckduckgo_search") from e

try:
    from duckduckgo_search.exceptions import RateLimitException as _DDGRateLimit
except Exception:
    class _DDGRateLimit(Exception):
        pass
import random, time

# --- Pomocnicze wywołanie z wieloma wariantami API ---
def _ddg_images_try_all(ddgs_obj, query: str, max_results: int):
    """
    Przetestuj różne podpisy i backendy API, zwróć iterator wyników lub None.
    """
    attempts = []

    # 1) Najczęstsze w 'ddgs': argument pozycyjny + backend 'lite'
    attempts.append(("pos_lite",
        lambda: ddgs_obj.images(query, max_results=max_results, safesearch="off", region="wt-wt", backend="lite")
    ))
    # 2) 'ddgs' / 'duckduckgo_search' z keywords= + backend 'lite'
    attempts.append(("keywords_lite",
        lambda: ddgs_obj.images(keywords=query, max_results=max_results, safesearch="off", region="wt-wt", backend="lite")
    ))
    # 3) 'ddgs' z query= + backend 'lite'
    attempts.append(("query_lite",
        lambda: ddgs_obj.images(query=query, max_results=max_results, safesearch="off", region="wt-wt", backend="lite")
    ))
    # 4) To samo z backend 'html' (czasem lite zwraca pusto)
    attempts.append(("pos_html",
        lambda: ddgs_obj.images(query, max_results=max_results, safesearch="off", region="wt-wt", backend="html")
    ))
    attempts.append(("keywords_html",
        lambda: ddgs_obj.images(keywords=query, max_results=max_results, safesearch="off", region="wt-wt", backend="html")
    ))
    attempts.append(("query_html",
        lambda: ddgs_obj.images(query=query, max_results=max_results, safesearch="off", region="wt-wt", backend="html")
    ))

    last_err = None
    for name, fn in attempts:
        try:
            it = fn()
            # iteratory w tych bibliotekach często nie są listą – „upewnijmy się”, że coś zwracają
            first = None
            collected = []
            for r in it:
                first = r
                collected.append(r)
                break
            if first is not None:
                # Zwróć z powrotem pierwszy element i resztę jako jednolitą listę
                return collected + [x for x in it]
        except TypeError:
            # Nieobsługiwane parametry w tej wersji – próbujemy kolejny wariant
            continue
        except Exception as e:
            last_err = e
            continue

    # Jeśli dotąd nic nie wyszło, a był błąd – podnieś go; w przeciwnym razie zwróć pustą listę
    if last_err is not None:
        # „No results found” bywa tak naprawdę 403/ratelimit – wyżej obsłużymy retry
        raise last_err
    return []
# -------------------------
# Wyszukiwanie URL-i z retry/backoff
# -------------------------
def fetch_image_urls(query: str, max_results: int, max_tries: int = 6) -> list[str]:
    urls: list[str] = []
    backoff = 2.0  # sekundy
    for attempt in range(1, max_tries + 1):
        try:
            with _DDGS() as ddgs_obj:
                results = _ddg_images_try_all(ddgs_obj, query, max_results)
            # Wyciąganie URL-a obrazka (klucze bywają różne: 'image', 'thumbnail', 'url')
            for r in results:
                url = r.get("image") or r.get("url") or r.get("thumbnail")
                if url and url not in urls:
                    urls.append(url)
                    if len(urls) >= max_results:
                        break
            if urls:
                break  # sukces
            else:
                # Brak wyników – często to chwilowy problem backendu. Odczekaj i spróbuj ponownie.
                wait = min(backoff * random.uniform(0.9, 1.6), 60)
                print(f"[DDG] Brak wyników dla '{query}'. Próba {attempt}/{max_tries}. Czekam {wait:.1f}s...")
                time.sleep(wait)
                backoff *= 1.8
        except (_DDGRateLimit,) as e:
            wait = min(backoff * random.uniform(1.2, 2.0), 90)
            print(f"[DDG] Rate limit dla '{query}'. Próba {attempt}/{max_tries}. Czekam {wait:.1f}s...")
            time.sleep(wait)
            backoff *= 2.0
        except Exception as e:
            # Inne błędy sieciowe / 403 itp.
            wait = min(backoff * random.uniform(1.0, 1.8), 90)
            print(f"[DDG] Błąd ({type(e).__name__}: {e}) dla '{query}'. Próba {attempt}/{max_tries}. Czekam {wait:.1f}s...")
            time.sleep(wait)
            backoff *= 1.8

    return urls[:max_results]

# -------------------------
# Pobieranie obrazów dla klasy
# -------------------------
def download_images_for_class(obj_name: str, dest_dir: Path, max_n: int) -> int:
    ensure_dir(dest_dir)
    urls = fetch_image_urls(obj_name, max_n)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    saved = 0
    seen = set()

    print(f"[{obj_name}] Znalezione URL-e: {len(urls)}. Pobieram do {max_n} obrazów...")
    for url in tqdm(urls, desc=f"Pobieranie: {obj_name}", unit="img"):
        if url in seen:
            continue
        seen.add(url)

        try:
            resp = session.get(url, timeout=TIMEOUT, stream=True)
            ct = resp.headers.get("Content-Type", "")
            status = resp.status_code
            if status == 429:
                # doraźna obsługa limitu po stronie hosta obrazka
                resp.close()
                wait = random.uniform(2.0, 5.0)
                print(f"[{obj_name}] 429 od hosta obrazka. Czekam {wait:.1f}s i próbuję dalej...")
                time.sleep(wait)
                continue

            if status != 200 or not is_image_content(ct):
                resp.close()
                time.sleep(SLEEP_BETWEEN)
                continue

            content = resp.content
            resp.close()

            if not validate_image_bytes(content):
                time.sleep(SLEEP_BETWEEN)
                continue

            ext = guess_extension_from_url_or_type(url, ct)
            base = f"{safe_filename(obj_name)}_{saved:04d}{ext}"
            out_path = dest_dir / base
            with open(out_path, "wb") as f:
                f.write(content)

            saved += 1
            if saved >= max_n:
                break

        except Exception:
            # Pomijamy błędne URL-e / timeouty itp.
            pass
        finally:
            # Utrzymuj umiarkowane tempo pobierania, by nie generować własnych limitów
            time.sleep(SLEEP_BETWEEN + random.uniform(0.0, 0.4))

    print(f"[{obj_name}] Zapisano {saved} obrazów do: {dest_dir}")
    return saved

# -------------------------
# Podział na zbiory
# -------------------------
def split_and_move(raw_root: Path, dataset_root: Path, train_pct: float = 0.9):
    for class_dir in sorted([p for p in raw_root.iterdir() if p.is_dir()]):
        cls = class_dir.name
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]
        if not files:
            print(f"[{cls}] Brak obrazów do podziału, pomijam.")
            continue

        random.shuffle(files)
        n = len(files)
        n_train = math.floor(train_pct * n)
        remaining = n - n_train
        n_val = remaining // 2
        n_test = remaining - n_val  # reszta

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            target_dir = dataset_root / split_name / cls
            ensure_dir(target_dir)
            for src in split_files:
                dst = target_dir / src.name
                shutil.move(str(src), str(dst))

        print(f"[{cls}] Podział: train={n_train}, val={n_val}, test={n_test}")

# -------------------------
# Główna procedura
# -------------------------
def main():
    random.seed(42)

    # 1–3. Pobierz obrazy i uporządkuj je w katalogach nazwanych jak obiekty (w RAW_DIR)
    for obj in OBJECTS:
        class_dir = RAW_DIR / obj
        download_images_for_class(obj, class_dir, MAX_PER_CLASS)

    # 4. Podziel na zbiory i przenieś do dataset/{train,val,test}/{klasa}
    #split_and_move(RAW_DIR, DATASET_DIR, TRAIN_PCT)

    print("\nZakończono.")
    print(f"Zbiory znajdują się w: {DATASET_DIR.resolve()}")
    print("Struktura:")
    for split in SPLITS:
        print(f" - {DATASET_DIR / split} / <nazwa_klasy> / <obrazy>")

if __name__ == "__main__":
    main()
