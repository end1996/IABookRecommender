"""
normalize_authors.py — Utilidad de normalización de autores

Script independiente para normalizar autores existentes en la base de datos.
NO busca autores nuevos (eso es responsabilidad de book_data_enrichment.py).

Operaciones:
- Last, First → First Last
- Eliminación de credenciales (Dr., PhD, etc.)
- Eliminación de apodos entre comillas
- Title Case
- Deduplicación de autores repetidos
- Detección de entidades corporativas (editorial, no persona)

Uso:
    python -m src.normalize_authors                    # Producción
    python -m src.normalize_authors --dry-run          # Preview sin cambios
    python -m src.normalize_authors --dry-run --limit 50  # Preview de 50
    python -m src.normalize_authors --export-csv       # Guardar reporte en CSV
"""

import argparse
import csv
import os
import sys

# Fix para consolas de Windows que no soportan emoji/Unicode por defecto
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import create_engine, text
from tqdm import tqdm

from config.settings import DB_CONFIG

# Reutilizamos las funciones de normalización ya definidas
from src.book_data_enrichment import normalizar_autor, JUNK_AUTHORS

# Conexión a la base de datos
engine = create_engine(
    f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}",
    echo=False,
)


def normalizar_autores_db(dry_run=False, limit=None, export_csv=False):
    """Normaliza todos los autores existentes en la base de datos.

    Solo modifica autores que ya tienen valor (no NULL/vacío).
    Compara el valor actual con el normalizado y solo actualiza si cambió.
    """
    print("\n" + "=" * 60)
    print("🔧 Normalización de autores existentes")
    print("=" * 60)

    query = """
        SELECT id, sku, title, author
        FROM books
        WHERE author IS NOT NULL 
          AND TRIM(author) != ''
          AND UPPER(TRIM(author)) NOT IN ('NA', 'N/A', 'SIN AUTOR', '-')
    """
    if limit:
        query += f" LIMIT {int(limit)}"

    with engine.connect() as conn:
        rows = conn.execute(text(query)).fetchall()

    if not rows:
        print("   ✅ No hay libros con autor para normalizar.")
        return {"total": 0, "normalizados": 0}

    print(f"   → {len(rows)} libros con autor existente")

    cambios = []
    eliminados = []  # Autores que quedan vacíos después de normalizar (eran basura)

    for row in tqdm(rows, desc="Analizando autores"):
        book_id, sku, titulo, autor_actual = row
        autor_normalizado = normalizar_autor(autor_actual)

        original = str(autor_actual).strip()

        # normalizar_autor devuelve None si todo era basura
        if autor_normalizado is None:
            eliminados.append({
                "id": book_id,
                "sku": sku,
                "titulo": titulo,
                "autor_anterior": original,
                "razon": "Solo contenía basura/credenciales",
            })
            continue

        # Solo guardar si realmente cambió
        if autor_normalizado != original:
            cambios.append({
                "id": book_id,
                "sku": sku,
                "titulo": titulo,
                "autor_anterior": original,
                "autor_nuevo": autor_normalizado,
            })

    print(f"\n📊 Análisis completado:")
    print(f"   → {len(cambios)} autores necesitan normalización")
    print(f"   → {len(eliminados)} autores eran basura (quedarían vacíos)")
    print(f"   → {len(rows) - len(cambios) - len(eliminados)} ya están correctos")

    # Exportar CSV si se pidió
    if export_csv and (cambios or eliminados):
        csv_file = "autores_normalizacion_preview.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["SKU", "Titulo", "Autor Anterior", "Autor Nuevo", "Tipo"])
            for c in cambios:
                writer.writerow([c["sku"], c["titulo"], c["autor_anterior"], c["autor_nuevo"], "normalización"])
            for e in eliminados:
                writer.writerow([e["sku"], e["titulo"], e["autor_anterior"], "(vacío)", "eliminado"])
        print(f"\n📄 Reporte exportado a: {csv_file}")

    # Preview
    if cambios:
        print(f"\n{'🔍 DRY-RUN: ' if dry_run else ''}Cambios de normalización:")
        for c in cambios[:20]:
            print(f"   [{c['sku']}] \"{c['titulo'][:40]}\"")
            print(f"      ❌ {c['autor_anterior']}")
            print(f"      ✅ {c['autor_nuevo']}")
        if len(cambios) > 20:
            print(f"   ... y {len(cambios) - 20} más")

    if eliminados:
        print(f"\n⚠️ Autores que se limpiarían (quedan vacío):")
        for e in eliminados[:10]:
            print(f"   [{e['sku']}] \"{e['titulo'][:40]}\" → \"{e['autor_anterior']}\"")
        if len(eliminados) > 10:
            print(f"   ... y {len(eliminados) - 10} más")

    # Aplicar cambios (solo normalización, NO los eliminados — esos requieren revisión manual)
    if not dry_run and cambios:
        print(f"\n💾 Guardando {len(cambios)} autores normalizados...")
        with engine.begin() as conn:
            sql = text("UPDATE books SET author = :autor WHERE id = :id")
            lote = []
            for i, c in enumerate(cambios):
                lote.append({"autor": c["autor_nuevo"], "id": c["id"]})
                if (i + 1) % 50 == 0:
                    conn.execute(sql, lote)
                    lote.clear()
            if lote:
                conn.execute(sql, lote)
        print("   ✅ Autores normalizados correctamente.")
    elif not dry_run and not cambios:
        print("\n   ✅ No hay cambios que aplicar.")

    if eliminados and not dry_run:
        print(f"\n   ⚠️ {len(eliminados)} autores-basura encontrados pero NO eliminados automáticamente.")
        print(f"   → Revisalos en el CSV (--export-csv) y limpiálos manualmente si corresponde.")

    stats = {
        "total": len(rows),
        "normalizados": len(cambios),
        "basura_detectados": len(eliminados),
        "sin_cambios": len(rows) - len(cambios) - len(eliminados),
    }
    print(f"\n📊 Resumen: {stats['normalizados']} normalizados | "
          f"{stats['basura_detectados']} basura | "
          f"{stats['sin_cambios']} sin cambios / {stats['total']} total")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Normaliza autores existentes en la base de datos de libros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python -m src.normalize_authors --dry-run              # Preview
  python -m src.normalize_authors --dry-run --export-csv  # Preview + CSV
  python -m src.normalize_authors                         # Aplicar cambios
  python -m src.normalize_authors --limit 100             # Solo los primeros 100
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Modo preview: muestra cambios sin modificar la BD",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar a N libros (útil para pruebas)",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Exportar reporte de cambios a CSV",
    )
    args = parser.parse_args()

    print("🔧 Utilidad de normalización de autores")
    print(f"   Modo: {'DRY-RUN (sin cambios en BD)' if args.dry_run else 'PRODUCCIÓN (cambios reales)'}")
    if args.limit:
        print(f"   Límite: {args.limit} libros")

    normalizar_autores_db(
        dry_run=args.dry_run,
        limit=args.limit,
        export_csv=args.export_csv,
    )


if __name__ == "__main__":
    main()
