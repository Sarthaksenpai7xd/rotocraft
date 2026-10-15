"""Command Line Interface (CLI) for Rotocraft.

Provides intuitive CLI commands:
  - rotocraft segment: Segment a single image with point/box prompts
  - rotocraft track: Propagate masks across video frames
  - rotocraft auto: Automatic full-scene object segmentation
  - rotocraft preview: Visualize and display mask overlays
  - rotocraft models: Manage and download SAM 2 weights
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import click
import cv2
import numpy as np
from PIL import Image

from rotocraft import __version__
from rotocraft.mask_export import MaskExporter
from rotocraft.models import AVAILABLE_MODELS, ModelManager, get_device
from rotocraft.preview import create_comparison, overlay_mask, save_preview
from rotocraft.segmenter import RotocraftSegmenter
from rotocraft.video_io import VideoReader, VideoWriter

# Configure root logger for CLI
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rotocraft.cli")


def parse_coords(val: str) -> tuple[float, float]:
    """Parse 'x,y' string into a (float, float) coordinate pair."""
    try:
        parts = [float(x.strip()) for x in val.split(",")]
        if len(parts) != 2:
            raise ValueError
        return parts[0], parts[1]
    except Exception:
        raise click.BadParameter(
            f"Expected coordinate format 'x,y', got '{val}' (e.g. --point 450,320)"
        )


def parse_box_coords(val: str) -> tuple[float, float, float, float]:
    """Parse 'x1,y1,x2,y2' string into a 4-float bounding box tuple."""
    try:
        parts = [float(x.strip()) for x in val.split(",")]
        if len(parts) != 4:
            raise ValueError
        return parts[0], parts[1], parts[2], parts[3]
    except Exception:
        raise click.BadParameter(
            f"Expected box format 'x1,y1,x2,y2', got '{val}' (e.g. --box 100,100,500,450)"
        )


@click.group()
@click.version_option(version=__version__, prog_name="rotocraft")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose debug logs.")
def main(verbose: bool) -> None:
    """Rotocraft: AI-powered automatic rotoscoping tool for FilmCraft."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command("models")
@click.option("-l", "--list", "list_flag", is_flag=True, help="List available models and download status.")
@click.option("-d", "--download", type=str, default=None, help="Download weights for specified model.")
@click.option("--cache-dir", type=click.Path(), default=None, help="Custom model cache directory.")
def cmd_models(list_flag: bool, download: Optional[str], cache_dir: Optional[str]) -> None:
    """List or download SAM 2 model weights."""
    manager = ModelManager(cache_dir=cache_dir)

    if download:
        click.echo(f"Downloading model '{download}'...")
        try:
            path = manager.download_model(download)
            click.secho(f"✓ Model '{download}' downloaded successfully to: {path}", fg="green")
        except Exception as e:
            click.secho(f"✗ Failed to download model '{download}': {e}", fg="red", err=True)
            sys.exit(1)
        return

    # Default to listing models
    models = manager.list_models()
    click.echo("\nAvailable SAM 2 Models:")
    click.echo("-" * 80)
    for m in models:
        status = click.style("INSTALLED", fg="green") if m["downloaded"] else click.style("NOT DOWNLOADED", fg="yellow")
        aliases = f" (aliases: {', '.join(m['aliases'])})" if m["aliases"] else ""
        click.echo(f"• {click.style(m['name'], bold=True)}{aliases}")
        click.echo(f"    Status: {status} | Size: ~{m['size_mb']} MB")
        if m["downloaded"]:
            click.echo(f"    Path:   {m['path']}")
        click.echo(f"    Desc:   {m['description']}\n")
    click.echo(f"Models directory: {manager.cache_dir}\n")


@main.command("segment")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True, help="Path to input image.")
@click.option("-o", "--output", "output_path", type=click.Path(), required=True, help="Output directory or file path.")
@click.option("-p", "--point", "points", multiple=True, type=str, help="Foreground point prompt 'x,y' (repeatable).")
@click.option("--bg-point", "bg_points", multiple=True, type=str, help="Background point prompt 'x,y' (repeatable).")
@click.option("-b", "--box", "box_str", type=str, default=None, help="Bounding box prompt 'x1,y1,x2,y2'.")
@click.option("-m", "--model", default="small", help="Model size: tiny, small, base_plus, large.")
@click.option("-f", "--format", "output_format", type=click.Choice(["png", "json", "filmcraft"], case_sensitive=False), default="png", help="Output format.")
@click.option("-d", "--device", default=None, help="Compute device: cuda, mps, cpu.")
@click.option("--clip-id", default=None, help="FilmCraft clip ID for command generation.")
@click.option("--preview/--no-preview", default=True, help="Save annotated preview image.")
def cmd_segment(
    input_path: str,
    output_path: str,
    points: tuple[str, ...],
    bg_points: tuple[str, ...],
    box_str: Optional[str],
    model: str,
    output_format: str,
    device: Optional[str],
    clip_id: Optional[str],
    preview: bool,
) -> None:
    """Segment an object in a single image with interactive point and box prompts."""
    # Parse points
    parsed_pts = [parse_coords(p) for p in points]
    parsed_labels = [1] * len(parsed_pts)

    parsed_bg_pts = [parse_coords(p) for p in bg_points]
    parsed_pts.extend(parsed_bg_pts)
    parsed_labels.extend([0] * len(parsed_bg_pts))

    parsed_box = parse_box_coords(box_str) if box_str else None

    out_p = Path(output_path)
    out_dir = out_p if out_p.is_dir() or not out_p.suffix else out_p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Initializing Rotocraft ({model})...")
    segmenter = RotocraftSegmenter(model_name=model, device=device)

    click.echo(f"Segmenting {input_path} using {segmenter.active_backend} backend...")
    result = segmenter.segment_frame(
        image=input_path,
        points=parsed_pts if parsed_pts else None,
        labels=parsed_labels if parsed_pts else None,
        boxes=parsed_box,
    )

    primary_mask = result.primary_mask
    score = float(result.scores[0]) if len(result.scores) > 0 else 1.0
    click.echo(f"Segmentation complete (confidence: {score:.2%}).")

    # Export results
    fmt = output_format.lower()
    if fmt == "png":
        target_file = out_p if out_p.suffix.lower() == ".png" else (out_dir / "mask.png")
        MaskExporter.to_png_sequence(primary_mask, target_file.parent, prefix=target_file.stem)
        click.secho(f"✓ Mask exported to: {target_file}", fg="green")
    elif fmt == "json":
        target_file = out_p if out_p.suffix.lower() == ".json" else (out_dir / "mask.json")
        data = MaskExporter.to_filmcraft_json(primary_mask)
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        click.secho(f"✓ Mask JSON exported to: {target_file}", fg="green")
    elif fmt == "filmcraft":
        target_file = out_p if out_p.suffix.lower() == ".json" else (out_dir / "filmcraft_cmd.json")
        cmd = MaskExporter.to_filmcraft_command(primary_mask, clip_id=clip_id)
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(cmd, f, indent=2)
        click.secho(f"✓ FilmCraft engine command saved to: {target_file}", fg="green")

    # Generate preview if requested
    if preview:
        img_rgb = RotocraftSegmenter._prepare_image(input_path)
        prev_file = out_dir / "preview.png"
        save_preview(
            image=img_rgb,
            mask=primary_mask,
            points=parsed_pts,
            labels=parsed_labels,
            boxes=[parsed_box] if parsed_box else None,
            output_path=prev_file,
        )
        click.echo(f"Preview saved to: {prev_file}")


@main.command("track")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True, help="Path to input video.")
@click.option("-o", "--output", "output_path", type=click.Path(), required=True, help="Output directory for mask sequence.")
@click.option("-p", "--point", "points", multiple=True, type=str, help="Initial point prompt 'x,y' on start frame.")
@click.option("--bg-point", "bg_points", multiple=True, type=str, help="Initial background point 'x,y'.")
@click.option("-b", "--box", "box_str", type=str, default=None, help="Initial box prompt 'x1,y1,x2,y2'.")
@click.option("--start-frame", type=int, default=0, help="Starting frame number (default: 0).")
@click.option("--end-frame", type=int, default=None, help="Ending frame number (default: last).")
@click.option("-m", "--model", default="small", help="Model size: tiny, small, base_plus, large.")
@click.option("-f", "--format", "output_format", type=click.Choice(["png", "json", "filmcraft", "video"], case_sensitive=False), default="png", help="Output format.")
@click.option("-d", "--device", default=None, help="Compute device: cuda, mps, cpu.")
@click.option("--clip-id", default=None, help="FilmCraft clip ID.")
def cmd_track(
    input_path: str,
    output_path: str,
    points: tuple[str, ...],
    bg_points: tuple[str, ...],
    box_str: Optional[str],
    start_frame: int,
    end_frame: Optional[int],
    model: str,
    output_format: str,
    device: Optional[str],
    clip_id: Optional[str],
) -> None:
    """Propagate object masks across video frames using SAM 2's video predictor."""
    parsed_pts = [parse_coords(p) for p in points]
    parsed_labels = [1] * len(parsed_pts)

    parsed_bg_pts = [parse_coords(p) for p in bg_points]
    parsed_pts.extend(parsed_bg_pts)
    parsed_labels.extend([0] * len(parsed_bg_pts))

    parsed_box = parse_box_coords(box_str) if box_str else None

    # Construct prompt specification
    init_prompt: dict[str, Any] = {"obj_id": 1}
    if parsed_pts:
        init_prompt["points"] = parsed_pts
        init_prompt["labels"] = parsed_labels
    if parsed_box:
        init_prompt["box"] = parsed_box

    prompts = {start_frame: [init_prompt]}

    click.echo(f"Initializing Rotocraft video tracker ({model})...")
    segmenter = RotocraftSegmenter(model_name=model, device=device)

    click.echo(f"Propagating masks across {input_path}...")
    tracked = segmenter.segment_video(
        video_path=input_path,
        prompts=prompts,
        output_dir=output_path,
        start_frame=start_frame,
        end_frame=end_frame,
        output_format=output_format,
        clip_id=clip_id,
    )

    click.secho(
        f"✓ Successfully tracked {len(tracked)} frames! Results exported to: {output_path}",
        fg="green",
    )


@main.command("auto")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True, help="Input image path.")
@click.option("-o", "--output", "output_dir", type=click.Path(), required=True, help="Output directory.")
@click.option("--min-area", type=float, default=0.001, help="Minimum object area ratio (default: 0.001).")
@click.option("-m", "--model", default="small", help="Model size: tiny, small, base_plus, large.")
@click.option("-d", "--device", default=None, help="Compute device.")
@click.option("-f", "--format", "output_format", type=click.Choice(["png", "json"], case_sensitive=False), default="png", help="Output format.")
def cmd_auto(
    input_path: str,
    output_dir: str,
    min_area: float,
    model: str,
    device: Optional[str],
    output_format: str,
) -> None:
    """Automatically detect and segment all objects in an image."""
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    click.echo(f"Initializing automatic segmenter ({model})...")
    segmenter = RotocraftSegmenter(model_name=model, device=device)

    click.echo(f"Analyzing {input_path} for objects...")
    masks = segmenter.auto_segment(image=input_path, min_area=min_area)
    click.echo(f"Discovered {len(masks)} objects.")

    img_rgb = RotocraftSegmenter._prepare_image(input_path)
    combined_overlay = img_rgb.copy()

    for i, item in enumerate(masks):
        m = item["segmentation"]
        color = (
            int((37 * (i + 1)) % 255),
            int((97 * (i + 1)) % 255),
            int((179 * (i + 1)) % 255),
        )
        combined_overlay = overlay_mask(combined_overlay, m, color=color, alpha=0.4)

        if output_format == "png":
            file_path = out_p / f"object_{i:03d}.png"
            MaskExporter.to_png_sequence(m, out_p, prefix=f"object_{i:03d}_")
        elif output_format == "json":
            file_path = out_p / f"object_{i:03d}.json"
            p_dict = MaskExporter.to_filmcraft_path(m)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(p_dict, f, indent=2)

    overview_path = out_p / "all_objects_preview.png"
    Image.fromarray(combined_overlay).save(str(overview_path))
    click.secho(f"✓ Discovered {len(masks)} objects. Overview saved to {overview_path}", fg="green")


@main.command("preview")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), required=True, help="Base image path.")
@click.option("-k", "--mask", "mask_path", type=click.Path(exists=True), required=True, help="Mask file path (PNG or JSON).")
@click.option("-o", "--output", "output_path", type=click.Path(), default=None, help="Save preview to image file.")
def cmd_preview(input_path: str, mask_path: str, output_path: Optional[str]) -> None:
    """Render and inspect mask overlay visualization."""
    base_img = RotocraftSegmenter._prepare_image(input_path)
    m_path = Path(mask_path)

    if m_path.suffix.lower() == ".json":
        with open(m_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Rasterize FilmCraft vector vertices to preview mask
        h, w = base_img.shape[:2]
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        vertices = data.get("vertices", [])
        if vertices:
            pts = np.array([v["p"] for v in vertices], dtype=np.int32)
            cv2.fillPoly(mask_arr, [pts], 255)
    else:
        mask_img = cv2.imread(str(m_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise click.ClickException(f"Could not read mask image: {m_path}")
        mask_arr = mask_img > 127

    preview_img = overlay_mask(base_img, mask_arr, color=(0, 255, 128), alpha=0.5)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(preview_img).save(str(out_p))
        click.secho(f"✓ Preview saved to {out_p}", fg="green")
    else:
        # Attempt interactive GUI window; fallback if headless
        try:
            cv2.imshow("Rotocraft Mask Preview (Press any key to close)", cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            fallback_out = Path("preview.png")
            Image.fromarray(preview_img).save(str(fallback_out))
            click.echo(f"Display unavailable (headless). Saved preview to {fallback_out}")


@main.command("app")
@click.option("-p", "--port", type=int, default=8765, help="Port to run the studio server on (default 8765).")
@click.option("--no-browser", is_flag=True, help="Do not open the browser automatically.")
def cmd_app(port: int, no_browser: bool) -> None:
    """Launch the interactive Rotocraft Studio Web GUI."""
    from rotocraft.web_app import start_studio_server

    server = start_studio_server(port=port, open_browser=not no_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down Rotocraft Studio.")
        server.shutdown()


@main.command("ui")
@click.option("-p", "--port", type=int, default=8765, help="Port to run the studio server on (default 8765).")
@click.option("--no-browser", is_flag=True, help="Do not open the browser automatically.")
def cmd_ui(port: int, no_browser: bool) -> None:
    """Alias for 'rotocraft app'."""
    from rotocraft.web_app import start_studio_server

    server = start_studio_server(port=port, open_browser=not no_browser)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down Rotocraft Studio.")
        server.shutdown()


if __name__ == "__main__":
    main()

