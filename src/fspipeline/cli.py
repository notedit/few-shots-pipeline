"""CLI entry point using Click + Rich."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from .config import load_config
from .pipeline import run_pipeline, run_single_stage

console = Console()


@click.group()
def main():
    """Few-shot TTS training data extraction pipeline."""
    pass


@main.command()
@click.option("-v", "--video", required=True, type=click.Path(exists=True), help="Input video file")
@click.option(
    "-r", "--reference", required=True, type=click.Path(exists=True), help="Reference speaker audio"
)
@click.option(
    "-c", "--config", "config_path", type=click.Path(), default=None, help="Config YAML path"
)
@click.option("-o", "--output", default="output", help="Output directory")
@click.option("--from-stage", default=None, help="Start from a specific stage")
def run(
    video: str,
    reference: str,
    config_path: str | None,
    output: str,
    from_stage: str | None,
):
    """Run the full extraction pipeline."""
    config = load_config(Path(config_path) if config_path else None)
    config.output_dir = output

    console.print(f"[bold]Pipeline starting[/bold]")
    console.print(f"  Video:     {video}")
    console.print(f"  Reference: {reference}")
    console.print(f"  Output:    {output}")

    ctx = run_pipeline(
        video_path=Path(video),
        reference_audio_path=Path(reference),
        config=config,
        start_stage=from_stage,
    )

    valid = sum(1 for s in ctx.segments if s.is_valid)
    total = len(ctx.segments)
    console.print(f"\n[bold green]Done![/bold green] {valid}/{total} valid segments")


@main.command()
@click.argument("stage_name")
@click.option("-i", "--input", "input_path", type=click.Path(exists=True), help="Input file")
@click.option(
    "-c", "--config", "config_path", type=click.Path(), default=None, help="Config YAML path"
)
@click.option("-o", "--output", default="output", help="Output directory")
def stage(stage_name: str, input_path: str | None, config_path: str | None, output: str):
    """Run a single pipeline stage."""
    config = load_config(Path(config_path) if config_path else None)
    config.output_dir = output

    console.print(f"[bold]Running stage:[/bold] {stage_name}")
    run_single_stage(
        stage_name=stage_name,
        config=config,
        input_path=Path(input_path) if input_path else None,
    )
    console.print(f"[bold green]Stage complete![/bold green]")


if __name__ == "__main__":
    main()
