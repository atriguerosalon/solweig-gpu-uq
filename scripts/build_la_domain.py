"""Build SOLWEIG-GPU input rasters for a Los Angeles neighbourhood from public data.

Sources (all public, no account needed):
- Ground elevation: USGS 3DEP, National Map ImageServer (resampled to the grid).
- Buildings: LA County countywide building outlines (LARIAC4, 2014 imagery), HEIGHT in feet.
- Trees: Meta / WRI global 1 m canopy height v1 (Tolan et al. 2024), CC BY 4.0.
- Land cover: ESA WorldCover 2021, 10 m, CC BY 4.0, mapped to SOLWEIG classes.

Outputs Building_DSM.tif (ground + buildings, absolute), DEM.tif, Trees.tif (canopy
height above ground) and Landcover.tif (1 paved, 2 building, 5 grass, 6 bare soil, 7 water),
in UTM zone 11N, the same layout as the SOLWEIG-GPU sample data.

Usage:
    python scripts/build_la_domain.py --lat 34.2735 --lon -118.410 --size 1000 --res 2 --out data/la/pacoima
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()

EPSG = 32611  # WGS 84 / UTM 11N
DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
BLD_URL = "https://dpw.gis.lacounty.gov/dpw/rest/services/FEMAFloodzoneAdopted/MapServer/15/query"
CHM_ROOT = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"
WC_URL = "/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
FT = 0.3048
# ESA WorldCover class -> SOLWEIG land-cover code. Ground under trees is treated as grass.
WC_TO_SOLWEIG = {10: 5, 20: 5, 30: 5, 40: 5, 50: 1, 60: 6, 70: 6, 80: 7, 90: 5, 95: 5, 100: 5}


def fetch(url: str, params: dict | None = None, timeout: int = 300) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "solweig-sensitivity research script"})
    for attempt in range(5):  # DNS inside WSL fails now and then
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"[fetch] attempt {attempt + 1} failed: {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}")


def grid(lat: float, lon: float, size: int, res: float) -> tuple[float, float, float, float]:
    src, dst = osr.SpatialReference(), osr.SpatialReference()
    src.ImportFromEPSG(4326)
    dst.ImportFromEPSG(EPSG)
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    x, y, _ = osr.CoordinateTransformation(src, dst).TransformPoint(lon, lat)
    half = size * res / 2
    x0, y0 = round(x - half), round(y - half)
    return x0, y0, x0 + size * res, y0 + size * res


def warp_to_grid(src: str, dst: Path, bounds, res: float, resample: str, dtype=gdal.GDT_Float32) -> np.ndarray:
    ds = gdal.Warp(str(dst), src, dstSRS=f"EPSG:{EPSG}", outputBounds=bounds, xRes=res, yRes=res,
                   resampleAlg=resample, outputType=dtype, creationOptions=["COMPRESS=DEFLATE"])
    a = ds.ReadAsArray()
    ds = None
    return a


def dem(bounds, size: int, res: float, raw: Path) -> Path:
    out = raw / "dem_3dep.tif"
    if not out.exists():
        pad = 50  # metres, so the warp to the grid has data at the edges
        b = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)
        n = int((b[2] - b[0]) / res)
        out.write_bytes(fetch(DEM_URL, {
            "bbox": ",".join(map(str, b)), "bboxSR": EPSG, "imageSR": EPSG, "size": f"{n},{n}",
            "format": "tiff", "pixelType": "F32", "interpolation": "RSP_BilinearInterpolation", "f": "image"}))
    return out


def buildings(bounds, raw: Path) -> Path:
    out = raw / "buildings.geojson"
    if out.exists():
        return out
    feats, offset = [], 0
    while True:
        d = json.loads(fetch(BLD_URL, {
            "geometry": ",".join(map(str, bounds)), "geometryType": "esriGeometryEnvelope", "inSR": EPSG,
            "spatialRel": "esriSpatialRelIntersects", "outFields": "HEIGHT,ELEV,BLD_ID", "outSR": EPSG,
            "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000}))
        feats += d.get("features", [])
        print(f"[buildings] {len(feats)} so far")
        if not d.get("exceededTransferLimit") and len(d.get("features", [])) < 1000:
            break
        offset += 1000
    out.write_text(json.dumps({"type": "FeatureCollection", "features": feats,
                               "crs": {"type": "name", "properties": {"name": f"EPSG:{EPSG}"}}}))
    return out


def chm_tile(lat: float, lon: float, raw: Path) -> str:
    index = raw / "chm_tiles.geojson"
    if not index.exists():
        index.write_bytes(fetch(f"{CHM_ROOT}/tiles.geojson"))
    pt = ogr.CreateGeometryFromWkt(f"POINT ({lon} {lat})")
    for f in json.loads(index.read_text())["features"]:
        if ogr.CreateGeometryFromJson(json.dumps(f["geometry"])).Contains(pt):
            return f"/vsicurl/{CHM_ROOT}/chm/{f['properties']['tile']}.tif"
    raise RuntimeError("no canopy-height tile covers this point")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--res", type=float, default=2.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out, raw = args.out, args.out.parent / "raw"
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    bounds = grid(args.lat, args.lon, args.size, args.res)
    print(f"[grid] EPSG:{EPSG} bounds {bounds}")

    ground = warp_to_grid(str(dem(bounds, args.size, args.res, raw)), out / "DEM.tif", bounds, args.res, "bilinear")
    print(f"[dem] {ground.min():.1f} to {ground.max():.1f} m")

    # Building heights rasterised onto the grid (feet -> metres), added to the ground.
    height = gdal.GetDriverByName("MEM").Create("", args.size, args.size, 1, gdal.GDT_Float32)
    height.SetGeoTransform((bounds[0], args.res, 0, bounds[3], 0, -args.res))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)
    height.SetProjection(srs.ExportToWkt())
    gdal.Rasterize(height, str(buildings(bounds, raw)), attribute="HEIGHT", allTouched=False)
    bh = height.GetRasterBand(1).ReadAsArray() * FT
    bh[(bh < 2.0) | (bh > 300)] = 0.0  # drop sheds and bad records
    is_bld = bh > 0
    print(f"[buildings] {100 * is_bld.mean():.1f}% of pixels, median height {np.median(bh[is_bld]):.1f} m")

    # Canopy height: average the 1 m product to the grid; no trees on roofs, ignore < 2 m.
    tree = warp_to_grid(chm_tile(args.lat, args.lon, raw), raw / "chm.tif", bounds, args.res, "average")
    tree = np.where(is_bld | (tree < 2.0) | ~np.isfinite(tree), 0.0, tree).astype(np.float32)
    print(f"[trees] {100 * (tree > 0).mean():.1f}% of pixels, median height {np.median(tree[tree > 0]):.1f} m")

    lat_tile = int(np.floor(args.lat / 3) * 3)
    lon_tile = int(np.floor(args.lon / 3) * 3)
    wc_name = f"N{lat_tile:02d}W{-lon_tile:03d}" if lon_tile < 0 else f"N{lat_tile:02d}E{lon_tile:03d}"
    wc = warp_to_grid(WC_URL.format(tile=wc_name), raw / "worldcover.tif", bounds, args.res, "near", gdal.GDT_Byte)
    lc = np.vectorize(lambda v: WC_TO_SOLWEIG.get(int(v), 1))(wc).astype(np.float32)
    lc[is_bld] = 2
    classes, counts = np.unique(lc, return_counts=True)
    print("[landcover] " + ", ".join(f"{int(c)}: {100 * n / lc.size:.1f}%" for c, n in zip(classes, counts)))

    ref = gdal.Open(str(out / "DEM.tif"))
    for name, arr in [("Building_DSM", ground + bh), ("Trees", tree), ("Landcover", lc)]:
        ds = gdal.GetDriverByName("GTiff").CreateCopy(str(out / f"{name}.tif"), ref, options=["COMPRESS=DEFLATE"])
        ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
        ds = None
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
