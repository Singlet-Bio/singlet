"""Command-line interface for sc-geo package."""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    """Main entry point for sc-geo CLI."""
    parser = argparse.ArgumentParser(
        prog="sc-geo",
        description="Large-scale single-cell RNA-seq processing from GEO",
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Catalog commands
    # ═══════════════════════════════════════════════════════════════════════
    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Build or manage GEO catalogs",
    )
    catalog_subparsers = catalog_parser.add_subparsers(dest="catalog_command")
    
    # catalog build
    build_parser = catalog_subparsers.add_parser(
        "build",
        help="Build complete GEO catalog with metadata",
    )
    build_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output file (.csv, .parquet, or .json)",
    )
    build_parser.add_argument(
        "--query",
        type=str,
        help="Custom NCBI search query",
    )
    
    # catalog discover (quick discovery only)
    discover_parser = catalog_subparsers.add_parser(
        "discover",
        help="Quick series discovery (UIDs only, no metadata)",
    )
    discover_parser.add_argument(
        "--output",
        type=Path,
        default="discovery.json",
        help="Output JSON file (default: discovery.json)",
    )
    discover_parser.add_argument(
        "--query",
        type=str,
        help="Custom NCBI search query",
    )
    
    # catalog filter
    filter_parser = catalog_subparsers.add_parser(
        "filter",
        help="Filter an existing catalog",
    )
    filter_parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input catalog file",
    )
    filter_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output catalog file",
    )
    filter_parser.add_argument(
        "--organisms",
        nargs="+",
        help="Filter by organisms (e.g., 'Homo sapiens' 'Mus musculus')",
    )
    filter_parser.add_argument(
        "--min-samples",
        type=int,
        help="Minimum samples per series",
    )
    filter_parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples per series",
    )
    
    # catalog stats
    stats_parser = catalog_subparsers.add_parser(
        "stats",
        help="Show catalog statistics",
    )
    stats_parser.add_argument(
        "input",
        type=Path,
        help="Catalog file",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Process commands
    # ═══════════════════════════════════════════════════════════════════════
    process_parser = subparsers.add_parser(
        "process",
        help="Process single-cell samples",
    )
    process_parser.add_argument(
        "--gsm",
        type=str,
        help="GSM accession to process",
    )
    process_parser.add_argument(
        "--gse",
        type=str,
        help="GSE accession (process all samples in series)",
    )
    process_parser.add_argument(
        "--organism",
        type=str,
        required=True,
        help="Organism name (e.g., 'human', 'mouse')",
    )
    process_parser.add_argument(
        "--output",
        type=Path,
        help="Output base directory (default: from config)",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Index commands
    # ═══════════════════════════════════════════════════════════════════════
    index_parser = subparsers.add_parser(
        "index",
        help="Build or manage reference indices",
    )
    index_subparsers = index_parser.add_subparsers(dest="index_command")
    
    # index build
    index_build_parser = index_subparsers.add_parser(
        "build",
        help="Build reference index for an organism",
    )
    index_build_parser.add_argument(
        "organism",
        type=str,
        help="Organism name (e.g., 'human', 'mouse', 'rat')",
    )
    index_build_parser.add_argument(
        "--type",
        choices=["piscem"],
        default="piscem",
        help="Index type (default: piscem)",
    )
    index_build_parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists",
    )
    
    # index list
    index_list_parser = index_subparsers.add_parser(
        "list",
        help="List available indices",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Batch commands (SLURM)
    # ═══════════════════════════════════════════════════════════════════════
    batch_parser = subparsers.add_parser(
        "batch",
        help="Submit and monitor SLURM batch jobs",
    )
    batch_subparsers = batch_parser.add_subparsers(dest="batch_command")
    
    # batch submit
    batch_submit_parser = batch_subparsers.add_parser(
        "submit",
        help="Submit batch processing job",
    )
    batch_submit_parser.add_argument(
        "catalog",
        type=Path,
        help="Catalog file to process (.csv or .parquet)",
    )
    batch_submit_parser.add_argument(
        "--job-name",
        default="scgeo-batch",
        help="Job name (default: scgeo-batch)",
    )
    batch_submit_parser.add_argument(
        "--partition",
        choices=["cpu", "bigmem", "gpu"],
        default="cpu",
        help="SLURM partition (default: cpu)",
    )
    batch_submit_parser.add_argument(
        "--samples-per-batch",
        type=int,
        default=50,
        help="Samples per batch (default: 50)",
    )
    batch_submit_parser.add_argument(
        "--cpus",
        type=int,
        default=38,
        help="CPUs per task (default: 38)",
    )
    batch_submit_parser.add_argument(
        "--memory",
        default="128G",
        help="Memory per node (default: 128G)",
    )
    batch_submit_parser.add_argument(
        "--time",
        default="12:00:00",
        help="Time limit HH:MM:SS (default: 12:00:00)",
    )
    batch_submit_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max concurrent array tasks (default: 20)",
    )
    batch_submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate script without submitting",
    )
    
    # batch monitor
    batch_monitor_parser = batch_subparsers.add_parser(
        "monitor",
        help="Monitor running batch job",
    )
    batch_monitor_parser.add_argument(
        "job_id",
        type=str,
        help="SLURM job ID to monitor",
    )
    batch_monitor_parser.add_argument(
        "--batch-dir",
        type=Path,
        help="Batch directory (default: pipeline/batches)",
    )
    batch_monitor_parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="Refresh interval in seconds (default: 30)",
    )
    
    # batch list
    batch_list_parser = batch_subparsers.add_parser(
        "list",
        help="List all SLURM jobs",
    )
    
    # batch cancel
    batch_cancel_parser = batch_subparsers.add_parser(
        "cancel",
        help="Cancel a SLURM job",
    )
    batch_cancel_parser.add_argument(
        "job_id",
        type=str,
        help="SLURM job ID to cancel",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Config commands
    # ═══════════════════════════════════════════════════════════════════════
    config_parser = subparsers.add_parser(
        "config",
        help="Show or manage configuration",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Show current configuration",
    )
    config_parser.add_argument(
        "--save",
        type=Path,
        help="Save configuration to file",
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # Version command
    # ═══════════════════════════════════════════════════════════════════════
    version_parser = subparsers.add_parser(
        "version",
        help="Show version information",
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Catalog commands
    # ═══════════════════════════════════════════════════════════════════════
    if args.command == "catalog":
        if args.catalog_command == "build":
            from scgeo.catalog import build_catalog
            
            print(f"Building complete GEO catalog...")
            catalog = build_catalog(
                query=args.query,
                output_file=args.output,
                include_metadata=True,
            )
            
            print(f"\n✓ Catalog complete:")
            print(f"  - {len(catalog):,} samples")
            print(f"  - {catalog['gse_id'].nunique():,} series")
            print(f"  - Output: {args.output}")
            
        elif args.catalog_command == "discover":
            from scgeo.catalog import discover_single_cell_series
            
            result = discover_single_cell_series(
                query=args.query,
                output_file=str(args.output),
            )
            
            print(f"\n✓ Discovered {result['count']:,} single-cell GSE series")
            print(f"  Output: {args.output}")
            
        elif args.catalog_command == "filter":
            from scgeo.catalog import load_catalog, filter_catalog, save_catalog
            
            print(f"Loading catalog from {args.input}...")
            catalog = load_catalog(args.input)
            print(f"  Loaded: {len(catalog):,} samples")
            
            print(f"Applying filters...")
            filtered = filter_catalog(
                catalog,
                organisms=args.organisms,
                min_samples=args.min_samples,
                max_samples=args.max_samples,
            )
            
            save_catalog(filtered, args.output)
            print(f"\n✓ Filtered catalog:")
            print(f"  - {len(filtered):,} samples ({100*len(filtered)/len(catalog):.1f}% of original)")
            print(f"  - Output: {args.output}")
            
        elif args.catalog_command == "stats":
            from scgeo.catalog import load_catalog, get_catalog_stats
            import json
            
            catalog = load_catalog(args.input)
            stats = get_catalog_stats(catalog)
            
            print(f"\nCatalog Statistics: {args.input}")
            print("=" * 60)
            print(f"Total samples:  {stats['total_samples']:,}")
            print(f"Total series:   {stats['total_series']:,}")
            print(f"\nOrganisms (top 10):")
            for org, count in sorted(stats['organisms'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {org}: {count:,}")
            print(f"\nDate range:")
            print(f"  Earliest: {stats['date_range']['earliest']}")
            print(f"  Latest:   {stats['date_range']['latest']}")
            print(f"\nSamples per series:")
            print(f"  Mean:   {stats['samples_per_series']['mean']:.1f}")
            print(f"  Median: {stats['samples_per_series']['median']:.0f}")
            print(f"  Range:  {stats['samples_per_series']['min']} - {stats['samples_per_series']['max']}")
            
        else:
            catalog_parser.print_help()
            
    # ═══════════════════════════════════════════════════════════════════════
    # Process command
    # ═══════════════════════════════════════════════════════════════════════
    elif args.command == "process":
        from scgeo.pipeline import process_sample
        
        if not args.gsm:
            print("Error: --gsm is required for process command")
            sys.exit(1)
        
        print(f"Processing {args.gsm}...")
        result = process_sample(
            gsm_id=args.gsm,
            gse_id=args.gse or "unknown",
            organism=args.organism,
            output_base=args.output,
        )
        
        print(f"\n{'✓' if result.status == 'success' else '✗'} Processing {result.status}")
        if result.status == "success":
            print(f"  Protocol: {result.detection.protocol}")
            print(f"  Cells: {result.qc.n_cells}")
            print(f"  Genes/cell: {result.qc.median_genes_per_cell}")
            print(f"  Mapping: {result.quantification.mapping_rate:.2%}")
            print(f"  Time: {result.total_time_s:.1f}s")
        else:
            print(f"  Error: {result.error}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Index command
    # ═══════════════════════════════════════════════════════════════════════
    elif args.command == "index":
        if args.index_command == "build":
            from scgeo.indices import build_index
            
            print(f"Building index for {args.organism}...")
            try:
                index_path = build_index(
                    organism=args.organism,
                    index_type=args.type,
                    force_rebuild=args.force,
                )
                print(f"\n✓ Index built successfully")
                print(f"  Type: {args.type}")
                print(f"  Path: {index_path}")
            except Exception as e:
                print(f"\n✗ Failed to build index: {e}")
                sys.exit(1)
        
        elif args.index_command == "list":
            from scgeo.indices import list_available_indices
            
            indices = list_available_indices()
            
            if not indices:
                print("No indices found.")
                print("\nBuild indices with: sc-geo index build <organism>")
            else:
                print("\nAvailable Indices:")
                print("=" * 60)
                for organism, types in sorted(indices.items()):
                    print(f"{organism:30s} {', '.join(types)}")
                print(f"\nTotal: {len(indices)} organisms")
        
        else:
            index_parser.print_help()
    
    # ═══════════════════════════════════════════════════════════════════════
    # Batch command
    # ═══════════════════════════════════════════════════════════════════════
    elif args.command == "batch":
        if args.batch_command == "submit":
            from scgeo.slurm import submit_batch
            from scgeo.config import get_config
            
            config = get_config()
            
            print(f"Submitting batch job for {args.catalog}...")
            
            job = submit_batch(
                catalog=args.catalog,
                job_name=args.job_name,
                partition=args.partition,
                samples_per_batch=args.samples_per_batch,
                cpus=args.cpus,
                memory=args.memory,
                time=args.time,
                max_concurrent=args.max_concurrent,
                dry_run=args.dry_run,
            )
            
            if job:
                print(f"\n✓ Job submitted successfully")
                print(f"  Job ID:       {job.job_id}")
                print(f"  Job Name:     {job.job_name}")
                print(f"  Partition:    {job.partition}")
                if job.array_size:
                    print(f"  Array Size:   {job.array_size}")
                print(f"  Script:       {job.script_path}")
                print(f"\nMonitor with: sc-geo batch monitor {job.job_id}")
            else:
                print("✗ Job submission failed")
                sys.exit(1)
        
        elif args.batch_command == "monitor":
            from scgeo.slurm import monitor_job
            from scgeo.config import get_config
            
            config = get_config()
            
            # Determine batch directory
            if args.batch_dir:
                batch_dir = args.batch_dir
            else:
                batch_dir = config.paths.pipeline_dir / "batches"
            
            if not batch_dir.exists():
                print(f"Error: Batch directory not found: {batch_dir}")
                sys.exit(1)
            
            print(f"Monitoring job {args.job_id}...")
            print(f"Batch directory: {batch_dir}")
            print()
            
            monitor_job(
                job_id=args.job_id,
                batch_dir=batch_dir,
                refresh_interval=args.refresh,
            )
        
        elif args.batch_command == "list":
            from scgeo.slurm import list_jobs
            
            list_jobs()
        
        elif args.batch_command == "cancel":
            from scgeo.slurm import cancel_job
            
            if cancel_job(args.job_id):
                print(f"✓ Cancelled job {args.job_id}")
            else:
                print(f"✗ Failed to cancel job {args.job_id}")
                sys.exit(1)
        
        else:
            batch_parser.print_help()
    
    # ═══════════════════════════════════════════════════════════════════════
    # Config command
    # ═══════════════════════════════════════════════════════════════════════
    elif args.command == "config":
        from scgeo.config import get_config
        
        config = get_config()
        
        if args.save:
            config.save(args.save)
            print(f"✓ Configuration saved to {args.save}")
        
        print("\nsc-geo Configuration")
        print("=" * 60)
        print(f"Project base:     {config.paths.project_base}")
        print(f"Catalog dir:      {config.paths.catalog_dir}")
        print(f"Pipeline dir:     {config.paths.pipeline_dir}")
        print(f"Index dir:        {config.paths.index_dir}")
        print(f"\nSupported species: {len(config.species_ref)}")
        print(f"Download segments: {config.download.segments}")
        print(f"QC min mapping:    {config.qc.min_mapping_rate}")
        print(f"simpleaf threads:  {config.resources.simpleaf_threads}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Version command
    # ═══════════════════════════════════════════════════════════════════════
    elif args.command == "version":
        from scgeo import __version__, __author__, __email__
        print(f"sc-geo version {__version__}")
        print(f"Author: {__author__} <{__email__}>")
        print(f"License: MIT")
        
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
