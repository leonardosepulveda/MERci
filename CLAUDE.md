# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains Python tooling for MERFISH (Multiplexed Error-Robust Fluorescence In Situ Hybridization) spatial transcriptomics experiments — specifically acquisition planning and online quality-control analysis during imaging. The primary package is `MERci/`.

## Environment setup

```bash
# One-time, per computer (uses mamba from Miniforge)
mamba env create -f environment.yml

# Activate and launch notebooks
mamba activate merci_env
jupyter lab
```

Open notebooks from their folders under `MERci/notebooks/` (`prepare_imaging/`, `analysis/`, `misc/`) in the JupyterLab file browser so that `SAMPLE_DIR` is auto-detected correctly.

## Deployment model

This repo is cloned directly into each new experiment folder (`SAMPLE_DIR`) as `SAMPLE_DIR/MERci/`. No `pip install` is needed — dependencies come from the `merci_env` conda environment. Most notebooks live two levels under the repo root (e.g. `MERci/notebooks/analysis/`), so they resolve paths as:

```python
MERCI_DIR  = Path(os.getcwd()).parent.parent   # MERci/
SAMPLE_DIR = MERCI_DIR.parent                   # experiment root
sys.path.insert(0, str(MERCI_DIR / "src"))      # MERci/src
```

The `prepare_imaging/` notebooks live **three** levels under the repo root
(`MERci/notebooks/prepare_imaging/<variant>/`, where `<variant>` is
`reference`), so they use
`MERCI_DIR = Path(os.getcwd()).parent.parent.parent`.

**Exceptions — `tumor` and `lineage_tracing`** are each split into two acquisition
types, one full copy of the six notebooks per type: `tumor/epi/` and `tumor/disk/`
(epifluorescence vs. spinning-disk confocal, same single-tissue-section layout);
`lineage_tracing/merfish/` and `lineage_tracing/lineage/` (MERFISH readout vs.
lineage-barcode readout, same multi-tissue layout). Those notebooks live **four**
levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`.

## Experiment folder layout

```
SAMPLE_DIR/          (the experiment root, e.g. D:\experiments\my_sample\)
  MERci/             ← clone of this repo
  positions/         ← boundaries/manual/ (hand-drawn boundary_positions*.txt, hole*.txt),
                        boundaries/from_mosaic/ (same files, auto-derived by prepare_imaging/02's
                        02_create_boundary_from_mosaic.ipynb) -- resolve_boundaries_source_dir picks
                        whichever has files (preferring from_mosaic); positions_{SAMPLE_NAME}.txt
                        (from prepare_imaging/02's 02_create_positions_from_boundaries.ipynb)
  metadata/          ← frame_table_*.csv, shutter_sequence_*.png (prepare_imaging/01),
                        round_info.csv, round_bit_color_map.csv (prepare_imaging/03),
                        data_organization_*.csv (prepare_imaging/05),
                        experiment_info.yaml (prepare_imaging/06)
  settings/          ← hal-config-*.xml, shutter-*.xml (prepare_imaging/01), dave-*.xml (prepare_imaging/04)
  data/              ← raw image files; exact subfolder structure defined by the `dir`
                        column in round_info.csv (written by HAL during acquisition)
  analysis/          ← thumbnails/, stats/, histograms/, mosaics/, done/
                        (produced by the analysis/01 and analysis/02 schedulers)
  merlin/            ← per-experiment MERlin config/run files (prepare_imaging/07):
                        analysis/merlin_analysis_{SAMPLE_NAME}.json,
                        snakemake/{parameters,cluster_resource_allocation}_{SAMPLE_NAME}.json,
                        slurm/submit/merlin_slurm_{SAMPLE_NAME}.sh
                        (replaces the old shared ~/Software/merfish-parameters/ location)
```

## Package layout

```
src/MERci/
  common/
    config.py       # ExperimentConfig dataclass — all paths and tunable parameters
    metadata.py     # ExperimentMetadata — parses round_info.csv + positions.txt
    io.py           # read_dax/zarr/tiff/image, parse_inf, get_dax_shape, load_round_info, load_positions,
                    # save_positions_array, discover_image_files, path_mtime (effective last-write mtime
                    # of a file OR a directory store -- max mtime of its contents, zarr-aware, since a
                    # directory's own mtime doesn't reliably update when a chunk file nested inside it is
                    # written; used by misc/measure_tissue_thickness.ipynb to measure real inter-FOV
                    # acquisition timing from file-write timestamps)
                    # + selective per-frame reading (only the requested frame_indices, real partial I/O
                    #   where the format allows it -- zarr fancy-indexing, dax direct byte-offset seeking,
                    #   tifffile's own key= selection -- rather than reading the whole stack and discarding
                    #   most of it): read_dax_frames, read_zarr_frames, read_tiff_frames, read_image_frames
                    #   (format-agnostic dispatcher, mirrors read_image) -- each reads every requested frame
                    #   eagerly, built on a lazy iter_dax_frames/iter_zarr_frames/iter_tiff_frames/
                    #   iter_image_frames counterpart that yields (frame_idx, frame) one at a time, so a
                    #   caller that might stop partway through frame_indices (e.g. a sequential scan with a
                    #   per-frame stopping condition -- see analysis.fov.measure_tissue_ntp_profile) never
                    #   reads/decodes the frames after where it actually stopped. iter_tiff_frames uses
                    #   TiffFile.asarray(key=idx, series=None), NOT tf.pages[idx] -- a stack tifffile.imwrite
                    #   writes from one (n_frames, H, W) array is normally stored as a single IFD holding
                    #   every frame (confirmed directly), not one page per frame, so tf.pages[idx] silently
                    #   reads the wrong data for exactly the files this package writes; omitting series=None
                    #   resolves a different, also-incorrect code path (confirmed directly). Note: tifffile's
                    #   key= selection has a confirmed quirk with very small (<5-frame) stacks -- raises
                    #   IndexError even via plain tifffile.imread(path, key=...) -- pre-existing, format-
                    #   library-level, not something this package can fix; never hit in practice since real
                    #   acquisitions are always well above 5 frames.
    experiment_info.py  # ExperimentInfo (core fields + extra dict, mirrors SeriesInfo.extra_meta's
                    #   pattern for the bc/lt/mf master-CSV schemas), save/load_experiment_info (flat
                    #   YAML round-trip), collect_experiment_info (batch-read many experiment_info.yaml
                    #   files into one outer-joined DataFrame, ready to append to a master CSV) +
                    #   resolve_sample_identity(merci_dir) -- true (sample_name, imaging_dir) from
                    #   folder structure alone: if MERCI_DIR's parent folder name is one of the fixed
                    #   acquisition-type subfolder tokens ("merfish"/"lineage"/"epi"/"disk"), that's the
                    #   split layout (true id one level further up); otherwise flat (parent folder name
                    #   IS the id). Used by every notebook from 02 onward for ALL local file naming
                    #   (positions_*.txt, dave-*.xml, data_organization_*.csv, merlin/fishtank script
                    #   filenames, …) as well as the cluster-facing DATA_HOME/MERLIN_HOME/FOLDER_NAME
                    #   (notebook 06) and resolve_cluster_sample_dir (notebook 07) -- one consistent
                    #   SAMPLE_NAME everywhere, never SAMPLE_DIR.name (which is only the acquisition's
                    #   own local folder tag, e.g. "merfish", once split into sibling subfolders)
  acquisition/
    configs.py      # get_frame_table, get_transit_frame_table (N blank frames), get_color_sequence_name
                    # (underscore-joined tokens, e.g. blkf5_488f2_560f25_650f25_750f25),
                    # get_color_to_channel_dict, create_shutter_file (per-colour power via power={nm:power}),
                    # create_hal_config, format_z_offsets_from_frame_table,
                    # resolve_power / power_dict_to_channel_list (colour->power, channel-ordered HAL default_power),
                    # naming rule: sequence_stem + hal_config_filename / shutter_filename / frame_table_filename
                    # (stem = "{kind}-{name}", kind in bits/cells/transit; hyphens delimit the prefix)
                    # + read_hal_flip_vertical, find_frame_table_for_hal_config, get_color_frame_indices
                    # + reconstruct_frame_table (inverse: hal+shutter XML -> frame table) and its parsers
                    #   read_shutter_reference, parse_z_offsets, parse_shutter_events
    positions.py    # create_grid_positions, generate_scanning_path, filter_scanning_path, close_scanning_path,
                    # load_hole_polygons (also reassembles hole{n}_island{m}.txt companions -- see mosaic.py --
                    #   into interior rings of that hole's Polygon, via Polygon(coords, holes=[...])), get_path_stats
                    # + multi-tissue: discover_boundary_files (auto multi/single/legacy), load_boundary_polygon,
                    #   create_transit_path (A->B, ~2x step), build_boundary_path (per-boundary pipeline), BoundarySpec,
                    #   has_boundary_files / resolve_boundary_dir (fall back to data/positions/examples when empty)
                    # + resolve_boundaries_source_dir(positions_dir, source=None) -- resolves
                    #   positions/boundaries/{manual,from_mosaic}/: with source=None, auto-picks whichever has
                    #   files (preferring from_mosaic), so notebooks 02 (FOV-grid generator) and 03
                    #   (round_info) always agree on which boundary set to use without passing state between them
    mosaic.py       # derive boundary_positions*.txt/hole*.txt automatically from a Steve low-mag mosaic
                    #   instead of drawing them by hand, writing to positions/boundaries/from_mosaic/ --
                    #   load_steve_mosaic (reads a Steve .msc manifest + its .stv tile pickles; each tile already
                    #   carries its own stage position + pixel size + zvalue (Steve's own display stacking order,
                    #   confirmed to increase monotonically with acquisition order), recovered from the tile's
                    #   own x_um/x_pix ratio and magnification, not a hard-coded coord.Point.pixels_to_um or the
                    #   .msc file's own rounded objective line) + filter_tiles_by_objective (kept for explicit
                    #   manual exclusion, e.g. rejecting bad tiles -- NOT applied by default any more, since
                    #   assemble_mosaic_canvas now handles mixed objectives/overlapping tiles directly),
                    #   assemble_mosaic_canvas (pastes tiles into one flattened image in stage-micron coords at
                    #   a single working_pixel_um resolution -- each tile is independently downsampled from its
                    #   own real pixel size, so tiles shot at different objectives/exposures composite together
                    #   without error; tiles are painted in ascending zvalue order with plain overwrite (not
                    #   averaging), so where N tiles overlap a pixel, the pixel comes from whichever tile is
                    #   physically on top -- e.g. a handful of 60x alignment FOVs shot over part of a 10x scan),
                    #   plot_tile_intensity_histograms (every tile's log-space intensity histogram overlaid as
                    #   thin gray lines, plus a solid combined histogram pooling every tile's pixels -- when that
                    #   combined histogram is clearly bimodal, _estimate_bimodal_threshold finds the valley
                    #   between its two most prominent peaks (scipy.signal.find_peaks) and returns it in linear
                    #   intensity units, drawn as a labelled vertical line and returned alongside the Axes, so
                    #   the notebook can seed THRESHOLD with it directly instead of starting from Otsu, which can
                    #   be biased toward the dominant class when one vastly outnumbers the other in pixel count
                    #   -- run this on tiles from one objective only (the majority one) when the mosaic mixes
                    #   objectives/exposures, since pooling differently-exposed tiles adds spurious extra modes
                    #   to the histogram and corrupts the valley estimate), segment_mosaic_tissue (smooth ->
                    #   threshold (Otsu default, or a fixed value read off the histogram) -> morphological
                    #   close/open/dilate-margin -> fill_holes -> per-component marching-squares contours ->
                    #   simplify; the smoothing+morphology is needed because a single global threshold on the
                    #   raw canvas fragments one tissue mass into hundreds of tiny disjoint specks from
                    #   illumination vignetting/tile seams. A labeled component can itself be a donut/annulus --
                    #   a hole with a real island of tissue inside it -- so per component every marching-squares
                    #   ring is inspected: the largest-area ring is the exterior, and any other ring above
                    #   min_island_area_um2 becomes an interior ring of a Polygon(exterior, holes=[...]) instead
                    #   of being discarded), plot_mosaic_segmentation (tissue/hole overlay, now also drawing each
                    #   hole's interior/island rings dashed, for the notebook's interactive threshold-tuning
                    #   review step), save_boundary_from_mosaic (writes in the exact convention
                    #   positions.discover_boundary_files/load_hole_polygons already expect -- legacy
                    #   boundary_positions.txt for one detected piece, boundary_positions_{b}.txt for several
                    #   disjoint pieces; holes are global, same as the rest of the pipeline; a hole with an
                    #   island also writes hole{n}_island{m}.txt companion files, one per interior ring)
    alignment.py    # cross-microscope FOV transfer: load_boundary_polygon, fit_isotropic_alignment
                    # (centroid/area init + IoU refinement, optional x/y axis flips), polygon_iou,
                    # AlignmentResult (scale + translation + flip_x/flip_y);
                    # bead-drift refinement: bead_frame_indices, select_bead_frame, extract_bead_frames
                    # (write bead-only .tiff), apply_orientation (inter-scope flip/transpose/rot to
                    # reconcile camera handedness HAL flip flags miss), phase_drift (skimage
                    # phase_cross_correlation, à la fishtank), compute_fov_drifts (mov_orient param);
                    # modality-robust path: detect_beads, register_point_translation (consensus
                    # voting + inlier-fraction score), compute_fov_drifts_beads
    dave.py         # create_round_info (positions_txt= derives the series pattern's FOV zero-pad
                    #   width from the real FOV count via fov_pad_width/count_positions -- e.g. 150
                    #   FOVs -> {fov:03d}, 1036 FOVs -> {fov:04d}; omitting it warns and falls back to
                    #   a fixed 3-digit width, since a hardcoded width silently stops matching real
                    #   files once FOV count crosses a digit boundary it didn't anticipate -- the
                    #   exact bug found in a live 1036-FOV round_info.csv stuck at 3 digits),
                    #   create_round_info_multitissue (per (round,segment) rows + positions_file/
                    #   tissue/segment/fov_pad cols; fov_pad likewise derived per boundary/transit
                    #   group via the same fov_pad_width, no artificial floor -- both this and
                    #   create_round_info share that one pad-width function now); both take
                    #   data_drives= to round-robin hyb rounds across physical drives),
                    #   rebase_on_drive/normalize_drive_root (public -- re-root a path onto a
                    #   different drive, preserving its subpath, e.g. G:\Leonardo\sample ->
                    #   V:\Leonardo\sample; used internally for round-robin hyb rounds AND by
                    #   06_transfer_to_nas.ipynb to derive TRANSFER_DEST from just a destination
                    #   drive, so the NAS path never has to be retyped by hand),
                    #   create_data_drive_skeleton (pre-creates each
                    #   hyb's H## folder only on the drive it's actually assigned to), create_dave_config
                    #   (positions_dir= enables per-segment loops; every loop gets its own identically-
                    #   named loop_variable even when several rounds share one positions file -- Dave's
                    #   real v2Generator indexes a loop by its own name, so a movie's <variable_entry>
                    #   cannot alias a differently-named shared loop_variable declared elsewhere),
                    #   dave_config_filename (single source of truth for the dave-{mic}-{N}hybs-{name}.xml
                    #   name, shared by the notebook that writes it and the one that later annotates it --
                    #   avoids globbing settings/dave-*.xml and picking the wrong file when two acquisitions
                    #   with different hyb counts share one settings/ folder), annotate_dave_with_round_info,
                    #   series_to_movie_name, get_hal_frame_count
    kilroy.py       # load_kilroy_protocols, find_kilroy_config (MF2 fallback),
                    # KilroyProtocolResolver — resolve dave fluidic steps to real Kilroy protocol names.
                    # + protocol/command consistency: load_kilroy_commands, iter_protocol_references,
                    #   check_kilroy_consistency (flag protocol steps naming an undefined valve/pump command,
                    #   with fuzzy-matched suggestions), format_consistency_report, fix_kilroy_consistency
                    #   (apply confirmed name fixes in place, backup *.bak, preserve CRLF + ISO-8859-1)
    data_organization.py  # create_data_organization
    merlin_config.py # MERlin input/config-file generation, writing to SAMPLE_DIR/merlin/ instead of
                    #   the old shared ~/Software/merfish-parameters/ location. Every schema verified
                    #   against real 2026 templates/outputs (see MERci/data/configs/merlin/):
                    #   create_microscope_parameters_json, create_codebook_csv (bit_names + gene/
                    #   Blank-N barcode rows; bit_names derivable from round_bit_color_map.csv +
                    #   readouts.csv, but the gene->barcode assignment itself must be supplied — a
                    #   wet-lab library-design input MERci cannot generate), create_cluster_resource_
                    #   allocation (duplicates a template's "Optimize01" block N times),
                    #   create_snakemake_parameters, create_slurm_submit_script (ports the current
                    #   live sbatch template), resolve_codebook_filename / resolve_microscope_
                    #   parameters_filename (lib_name/microscope -> filename dispatch only) +
                    #   MerlinAnalysisSpec / create_merlin_analysis_parameters — builds MERlin's
                    #   warp/optimize/decode/segment task-parameters JSON from a compact spec (which
                    #   steps to include: n_optimize_iterations, include_reporting,
                    #   include_segmentation + method), replacing copy+hand-edit of a prior
                    #   experiment's file entirely
    fishtank_config.py # fishtank input/config-file generation for lineage_tracing/lineage
                    #   experiments ONLY (analyzed with fishtank, not MERlin — every other
                    #   variant/acquisition-type uses merlin_config.py above), writing to
                    #   SAMPLE_DIR/fishtank/. Every schema/script verified against a real
                    #   reference experiment (see MERci/data/configs/fishtank/):
                    #   create_color_usage_csv (per-round/color target table — the round-tag
                    #   mapping is a manual, per-protocol choice, not derived from round_info.csv),
                    #   create_decoding_strategy_csv, resolve_fishtank_reference_dir (library-
                    #   version -> shared reference dir, dispatch only), create_fishtank_folder_
                    #   skeleton, copy_fishtank_reference_files + FishtankScriptsSpec /
                    #   create_fishtank_scripts — builds every fishtank run script (cellpose,
                    #   detect-spots, decode-spots, mosaics) from a compact, fully-overridable
                    #   spec (mirrors MerlinAnalysisSpec)
    display.py      # print_frame_table, display_xml (Jupyter helpers)
    cluster_submit.py # sbatch script generation + submission for cluster-side QC analysis
                    #   (07_cluster_submit_analysis.ipynb): build_fov_array_script/
                    #   build_round_mosaic_script (fishtank_config.py's _sbatch_header
                    #   conventions — FOV-parallel array jobs, not MERlin's single-job
                    #   convention), submit_sbatch/job_state/is_job_active (subprocess
                    #   wrappers around sbatch/sacct, never raising — a submission hiccup
                    #   just logs and returns None)
  analysis/
    fov.py          # create_thumbnail(s), measure_stats, get_histogram, load_stats, load_histogram,
                    # analyze_file (top-level per-FOV worker: read once + all analyses + sentinel),
                    # compute_histogram_only (same picklable-top-level-function convention as
                    # analyze_file, for a ProcessPoolExecutor worker, but histogram only — skips
                    # thumbnails/stats for callers outside the standard pipeline that don't need them),
                    # compute_channel_counters (reads every z-plane in a given channel's frame range via
                    # io.iter_image_frames — still far fewer frames than the whole multi-color stack — and
                    # builds an EXACT, bin-width-1 histogram of each frame: a true Counter over observed
                    # pixel intensities, stored SPARSELY as (values, counts) pairs via numpy.unique rather
                    # than a dense fixed-width array — used by misc/measure_tissue_thickness.ipynb, which
                    # needs every intensity value's exact count, not a lossy fixed-bin approximation, so it
                    # can derive a mean, a percentile, a re-binned view, and a true-pixel count for any
                    # threshold from ONE cached read instead of recomputing/re-reading pixels for each),
                    # save_channel_counters/load_channel_counters (persist/reload — ragged per-z arrays,
                    # stored as numpy object arrays, needing allow_pickle=True to read back),
                    # counter_mean/counter_percentile/rebin_counter/ntp_from_counter (derive a mean /
                    # percentile / re-binned histogram over arbitrary bin_edges / true-pixel count from a
                    # (values, counts) Counter — pure arithmetic, no raw pixel re-read), 
                    # ntp_profile_from_counters (per-z true-pixel-count profile purely from a
                    # compute_channel_counters() result — no disk read at all once Counters are cached),
                    # _summarize_ntp_profile (shared helper: given every z's true-pixel count, returns
                    # z_first_um/z_last_um — shallowest/deepest z with signal, either None if none passed —
                    # and is_contiguous, False if signal turned off and back on somewhere in between; real
                    # data showed tissue signal isn't always monotonic with depth, so every z must be
                    # counted rather than stopping at the first failure) — FOV-level analysis
    round.py        # create_mosaic, load_thumbnails_for_round — round-level mosaic
    stage_z.py      # stage-z drift QC: off_path_for/read_off_file (HAL's per-movie ``.off``
                    #   focus-lock sidecar — same directory/stem convention as ``.inf``, whitespace-
                    #   delimited, one row per frame, column ``stage-z``), summarize_stage_z (first/
                    #   min/max + all_same, since the focus lock should hold stage-z constant for a
                    #   whole stack), update_stage_z_cache (extends an on-disk CSV cache with only
                    #   not-yet-read (round, FOV, series) combinations, so a many-thousand-FOV
                    #   experiment's ``.off`` files are each read at most once), round_label/
                    #   assign_x_positions (continuous cells→hyb01→hyb02→... FOV ordering for
                    #   plotting drift over the whole acquisition) — see notebooks/analysis/08
    spot_localization.py  # bead detection / 3D Gaussian fitting + PSF simulation (detect_beads_2d,
                          # localize_beads_in_file, match_beads_across_colors, simulate_multicolor_stack, …)
    cli_analyze_fov.py       # standalone SLURM-array-task script (not imported by anything else in
                    #   the package) — self-locates its own src/ root from __file__, so no pip
                    #   install is needed on the cluster; runs analyze_file for one manifest-line
                    #   FOV per array task, via the same build_fov_task_kwargs scheduler.py uses
    cli_build_round_mosaic.py # same self-locating convention; runs build_round_mosaics for
                    #   --round-id (or a manifest of round ids for an array job)
  state.py          # ExperimentStateMonitor — detects imaging vs. fluidics phases by watching file mtimes
  progress.py       # ProgressTracker — sentinel files for fov_done, round_done, round_transferred,
                    #   fov_transferred (per-FOV, verified-copy transfer — distinct from fov_done, which
                    #   is local QC analysis completion, not a transfer state), fov_submitted/
                    #   round_mosaic_submitted (cluster SLURM bookkeeping, hold the job id)
  progress_display.py # ProgressReporter — reusable, dependency-free live console/notebook progress +
                    #   ETA display for any long-running per-item loop (n/total, percent, a text bar,
                    #   elapsed, ETA extrapolated from the average per-item rate seen so far); wrap an
                    #   iterable directly (`for x in ProgressReporter(len(items), "label").wrap(items)`)
                    #   or drive update()/done() manually. Distinct from progress.py's ProgressTracker,
                    #   which persists completion via on-disk sentinels across separate runs — this is
                    #   purely a live display for a loop running right now, no persistence. Not used by
                    #   TransferScheduler (which already has its own bespoke, more detailed copy/verify-
                    #   split timing — see scheduler.py) — meant for notebooks that don't have that
                    #   already, e.g. misc/measure_tissue_thickness.ipynb's histogram backfill loop.
  scheduler.py      # FOVScheduler (continuous, parallel process-pool), RoundScheduler, TransferScheduler
                    #   (round_robin_drives-only: transfers once a round is fully written, decoupled
                    #   from local analysis, ONE FOV AT A TIME, each verified (verify_method="hash"|"sample"|
                    #   "size" constructor param + verify_sample_bytes, see transfer.py's verify_copy) before
                    #   being marked transferred — a round is round_transferred only once every one of its FOVs has verified, at
                    #   which point (delete_source_after_verify=True, off by default — irreversible) its
                    #   entire source data directory is deleted; average_seconds_per_fov/
                    #   average_copy_seconds_per_fov/average_verify_seconds_per_fov track copy and verify
                    #   time in separate rolling deques (n_fov_rate_samples reports how many samples that
                    #   actually is right now, capped at 200 — not a claim that 200 already happened) so a
                    #   slow rate can be diagnosed as network-bound vs. verification-bound; sync_static_metadata()
                    #   handles the small, unchanging-once-acquisition-starts files (round_info.csv rewritten/
                    #   round_bit_color_map.csv/positions_*.txt/settings/*.xml) — call it once, not from
                    #   the tick loop, since re-syncing them every tick was pure overhead — see "Moving
                    #   QC analysis to a SLURM cluster" below),
                    #   ExperimentScheduler — main analysis loops; also exports the shared
                    #   build_fov_task_kwargs/build_round_mosaics/resolve_round_flip_y/
                    #   resolve_round_color_frame_indices/source_dirs_for_round functions the
                    #   schedulers and the cli_*.py cluster scripts both call
  transfer.py       # transfer_round (per-round → NAS, one background thread per round; destination is
                    #   dest_root/relative_to_data_root(src) — e.g. dest_root/data/hybs/H01 — NOT dest_root/
                    #   src.name, so a round-robin round transferred from its own physical drive (e.g.
                    #   D:/.../data/hybs/H01) still lands at the SAME logical data/... location a same-drive
                    #   round would, and TRANSFER_DEST ends up a full mirror of SAMPLE_DIR rather than a flat
                    #   dump of per-round folder names), relative_to_data_root (strips everything before a
                    #   path's own 'data' component — the shared logic behind that destination choice),
                    #   rewrite_round_info_dirs (copies round_info.csv rewriting its dir column's absolute,
                    #   drive-specific per-round paths to that same relative data/... form, so a cluster-side
                    #   ExperimentMetadata.load() reading the transferred copy resolves paths under ITS OWN
                    #   data_dir instead of a microscope-local drive letter that doesn't exist there — the
                    #   original microscope-side round_info.csv, which must keep absolute per-drive paths for
                    #   the live acquisition, is never touched), mirror_tree (incremental data_dir → 2nd
                    #   drive, background thread) / mirror_dir_sync (same copy logic, but synchronous — for a
                    #   notebook cell that wants to see a one-off folder sync finish, e.g. data/mosaic10x or
                    #   the static MERci/merlin/fishtank folders, before moving on) — robocopy/shutil either
                    #   way; sync_files (synchronous small-file copy, preserving relative path —
                    #   metadata/positions/settings).
                    #   Per-FOV verified transfer (used by TransferScheduler instead of transfer_round —
                    #   one bulk robocopy per round with no post-copy check): fov_associated_paths (every
                    #   file/dir sharing an image's stem — the store itself plus whatever same-stem
                    #   .inf/.off/.power/.xml sidecars HAL wrote, found by glob, not a fixed extension
                    #   list), copy_fov (copies them, preserving data/... structure), verify_copy(src, dst,
                    #   method="hash"|"sample"|"size", sample_bytes=...) — "hash": full SHA-256 of every
                    #   file, not just size/timestamp (all robocopy's own "up to date" check does, which
                    #   would not catch silent corruption); reads every byte TWICE (source + destination),
                    #   which can dominate total transfer time over a slow network share (measured
                    #   ~50-80 s/FOV on a 14504-FOV experiment on 2026-07-14/15, vs ~1 s/FOV for "size") —
                    #   "sample": SHA-256 of just the first+last sample_bytes of each file (whole file if
                    #   smaller), bounded I/O regardless of file size — a middle ground added 2026-07-15:
                    #   catches truncation with certainty and corruption within the sampled head/tail
                    #   windows, but can miss corruption confined to a large file's untouched middle —
                    #   "size": file size + exact same file set only, no content read at all, fastest,
                    #   still catches the realistic LAN failure mode (truncated/incomplete copy) but not a
                    #   same-size corruption; for a directory, recurses and also requires the exact same
                    #   set of relative paths on both sides.
                    #   transfer_fov(image_path, dest_root, verify_method=...) — copy_fov + verify_copy,
                    #   returns {verified, copy_seconds, verify_seconds} (timed separately so a slow FOV
                    #   can be diagnosed as copy-bound vs. verify-bound instead of one combined number),
                    #   delete_source_tree (rmtree/unlink — does no verification itself; the caller,
                    #   TransferScheduler, is the safety boundary and only ever calls this once every
                    #   FOV in a round has independently verified)
  visualization.py  # visualize_shutter_sequence, plot_fov_layout, plot_stats_over_rounds, display_mosaic
  disk_audit.py     # discover_sample_dirs, measure_folder (recursive size + file mtime range +
                    #   folder creation time), audit_disk_usage — scans {root}/{lab_member}/{sample_dir}
                    #   layouts on shared microscope-computer drives to find old/large data to clean up
notebooks/
  prepare_imaging/  # Pre-experiment notebooks (run in order), split into per-experiment variants:
    reference/       # canonical, fully-featured templates (keep up to date)
    tumor/           # single tissue section per coverslip; split by acquisition type:
      epi/           #   epifluorescence acquisition — full copy of the 8 notebooks
      disk/          #   spinning-disk confocal acquisition — full copy of the 8 notebooks
                     #   (both four levels deep -> MERCI_DIR = ...parent.parent.parent.parent)
    lineage_tracing/ # multiple tissue sections per coverslip; split by acquisition type:
      merfish/       #   MERFISH (codebook) acquisition — full copy of the 8 notebooks
      lineage/       #   lineage-barcode acquisition — full copy of the 8 notebooks, but with
                     #     DIFFERENT 05/07 notebooks (see below) — analyzed with fishtank, not
                     #     MERlin
                     #   (both four levels deep -> MERCI_DIR = ...parent.parent.parent.parent)
      # each of tumor/{epi,disk}/ and lineage_tracing/{merfish,lineage}/ contains:
      #   01_create_hal_config_and_shutters.ipynb     # imaging sequence, per-channel POWER, HAL/shutter for
      #                                                #   bits+cells, and a transit HAL config (blank frames)
      #   02_create_boundary_from_mosaic.ipynb        # OPTIONAL, independent of the notebook below: derive
      #                                                #   boundary_positions*.txt/hole*.txt automatically from
      #                                                #   a Steve low-mag mosaic instead of drawing them by
      #                                                #   hand; writes to positions/boundaries/from_mosaic/
      #   02_create_positions_from_boundaries.ipynb   # multi-boundary FOV grids + transit segments; per-segment /
      #                                                #   per-tissue positions files; creates data/ subfolders.
      #                                                #   Reads boundary files from positions/boundaries/manual/
      #                                                #   or positions/boundaries/from_mosaic/ (auto-picks
      #                                                #   whichever has files, preferring from_mosaic --
      #                                                #   resolve_boundaries_source_dir)
      #   03_create_round_info.ipynb                  # round-bit-color map (+ derives N_HYBS) + round_info.csv
      #   04_create_dave_config.ipynb                 # builds the Dave recipe XML from round_info.csv; every
      #                                                #   loop gets its own identically-named loop_variable
      #                                                #   (Dave requires the exact-name match, even when
      #                                                #   several rounds share one positions file)
      #   06_create_experiment_info.ipynb             # writes metadata/experiment_info.yaml (auto-fills what MERci
      #                                                #   already knows; biology/cluster-path fields left for the user)
      # tumor/{epi,disk}/ and lineage_tracing/merfish/ (MERlin-based) additionally have:
      #   05_create_data_organization.ipynb           # MERlin data-organization setup (transit-safe series pick);
      #                                                #   also annotates the notebook-04 Dave XML with bit info
      #   07_create_merlin_scripts.ipynb              # writes SAMPLE_DIR/merlin/ (analysis-parameters JSON via
      #                                                #   MerlinAnalysisSpec, snakemake/cluster-allocation JSON, slurm
      #                                                #   submit script); references the shared codebook/microscope
      #                                                #   files shipped in MERci/data/configs/merlin/ by path
      # lineage_tracing/lineage/ (fishtank-based) instead has:
      #   05_create_color_usage.ipynb                 # fishtank's color_usage (manual round-tag mapping,
      #                                                #   not derived from round_info.csv) + decoding_strategy
      #   07_create_fishtank_scripts.ipynb            # writes SAMPLE_DIR/fishtank/ (folder skeleton, shared
      #                                                #   reference files, every run script) via a
      #                                                #   FishtankScriptsSpec; reads the sibling merfish
      #                                                #   acquisition's data/positions (../merfish/, confirmed
      #                                                #   sample layout: <sample_id>/{merfish,lineage}/)
  analysis/         # Online-analysis notebooks (run during the experiment)
    01_fov_scheduler.ipynb                         # FOV-level scheduler (thumbnails, stats, histograms)
    02_round_scheduler.ipynb                       # round-level scheduler (mosaics, optional data transfer)
    03_view_mosaics.ipynb                          # display per-color mosaics as they are built
    04_view_intensity_stats.ipynb                  # plot per-frame intensity statistics over rounds
    05_batch_sample_review.ipynb                   # post-acquisition: verify a batch of experiments' analysis
                                                   #   is complete (backfill via analyze_file if not), then plot
                                                   #   per-round intensity/saturation comparisons across the batch —
                                                   #   the MERlin-independent half of the old cluster-side review
                                                   #   notebook (excludes anything needing MERlin's decoded output)
    06_transfer_to_nas.ipynb                       # round_robin_drives only: TransferScheduler continuously
                                                   #   copies each round's raw data ONE FOV AT A TIME (once fully
                                                   #   written, decoupled from any local QC), each verified per
                                                   #   VERIFY_METHOD ("hash": full SHA-256, reads every byte
                                                   #   twice, can dominate transfer time over a slow network
                                                   #   share; "sample": SHA-256 of just the first+last
                                                   #   VERIFY_SAMPLE_BYTES of each file, a middle ground; "size":
                                                   #   size + file-count only, fastest, still catches truncated
                                                   #   copies) before being marked transferred,
                                                   #   nested under TRANSFER_DEST exactly as it is locally (data/cells,
                                                   #   data/hybs/H01, ... -- see transfer.relative_to_data_root)
                                                   #   -- a round is round_transferred only once every FOV in it
                                                   #   has verified, at which point (DELETE_SOURCE_AFTER_
                                                   #   VERIFIED_TRANSFER, off by default) its source directory is
                                                   #   deleted; TRANSFER_DEST is derived from just a destination
                                                   #   drive (dave.rebase_on_drive) rather than typed by hand; a
                                                   #   one-time (not per-tick) section syncs round_info.csv
                                                   #   (rewritten, via transfer.rewrite_round_info_dirs) /
                                                   #   round_bit_color_map.csv/positions_*.txt/settings/*.xml plus
                                                   #   the equally-static data/mosaic10x/MERci/merlin/fishtank
                                                   #   folders, so TRANSFER_DEST ends up a full copy of
                                                   #   SAMPLE_DIR -- run on the microscope computer alongside
                                                   #   HAL/Dave, as the companion to moving QC analysis to a
                                                   #   cluster
    07_cluster_submit_analysis.ipynb               # run ON a cluster login/transfer node (after you've moved data
                                                   #   from the NAS to cluster storage yourself, e.g. Globus/
                                                   #   FileZilla): discovers pending FOVs/rounds and submits SLURM
                                                   #   array jobs (cli_analyze_fov.py / cli_build_round_mosaic.py,
                                                   #   via acquisition/cluster_submit.py) to do the QC analysis
                                                   #   01/02 would otherwise do locally -- 01/02 remain supported
                                                   #   for same_drive/mirror_drive projects; 06/07 are the
                                                   #   round_robin_drives + cluster-QC alternative
    08_stage_z_drift.ipynb                         # one-shot QC: reads each FOV's ``.off`` focus-lock
                                                   #   sidecar's stage-z column (analysis/stage_z.py),
                                                   #   caching results to analysis/stage_z_summary.csv so
                                                   #   re-runs only read newly-written ``.off`` files, then
                                                   #   scatter-plots the first-frame stage-z value across a
                                                   #   continuous cells→hyb01→hyb02→... FOV order to visualise
                                                   #   drift over the whole acquisition
  misc/             # Ad-hoc utilities
    MF2_60XSil1.3_zcorrection.ipynb                # z-correction helper for the MF2 60x silicone objective
    reconstruct_frame_table_from_configs.ipynb     # inverse of prepare_imaging/01: hal+shutter XML -> frame_table CSV
    align_fovs_across_microscopes.ipynb            # map FOV positions from one scope to another: Part 1 overlaps
                                                   #   tissue boundaries (isotropic scale+translation+optional
                                                   #   axis flips, no rotation); Part 2 refines per-FOV bead drift
                                                   #   via phase_cross_correlation. Inputs are explicit per-dir paths.
    extract_source_bead_frames.ipynb               # run at the source scope: write a compact per-FOV bead-only
                                                   #   .tiff (+ compact frame table) so only the bead frames move
                                                   #   to the NAS for align_fovs Part 2
    verify_kilroy_protocol_consistency.ipynb       # verify a Kilroy config's protocols only reference defined
                                                   #   valve/pump commands; fuzzy-suggest + (after confirm) rewrite
                                                   #   mismatches in place (backup to *.bak), via kilroy.py helpers
    audit_disk_usage.ipynb                         # scan a list of shared-drive roots laid out as
                                                   #   {root}/{lab_member}/{sample_dir}, measure each sample
                                                   #   folder's size + creation date (disk_audit.py), and display
                                                   #   it sorted oldest-first and largest-first — to find whose
                                                   #   data to ask to be cleared off a shared microscope computer
    measure_tissue_thickness.ipynb                 # per-FOV map of where (z, µm) real tissue signal
                                                   #   starts and ends, for one finished round (default:
                                                   #   cells) — a variable-thickness sample wastes imaging
                                                   #   time on z-planes with no tissue, at EITHER end of the
                                                   #   stack (real data showed some FOVs are blank at the top
                                                   #   of the imaged range and only pick up signal partway
                                                   #   down, not just "signal that eventually stops"). One
                                                   #   backfill step, sequential rather than process-pool-
                                                   #   parallelized (see the BrokenProcessPool note below): for
                                                   #   every FOV, read every z-plane of CHANNEL_NM (still far
                                                   #   fewer frames than the round's full multi-color stack)
                                                   #   and build an EXACT, bin-width-1 histogram of each frame
                                                   #   — a true Counter over observed pixel intensities
                                                   #   (analysis.fov.compute_channel_counters, stored sparsely
                                                   #   via numpy.unique, cached per FOV under analysis/
                                                   #   tissue_thickness_cache/channel_counters/ — deliberately
                                                   #   NOT the canonical tracker.histogram_path() routine
                                                   #   analysis uses, and with no fallback-reuse of an existing
                                                   #   full histogram either: FOVScheduler's own histograms are
                                                   #   fixed-width/lossy 512-bin ones, which can't be turned
                                                   #   back into an exact per-intensity Counter — writing a
                                                   #   partial histogram to that canonical path was also a real
                                                   #   bug hit and fixed earlier in development, since
                                                   #   analyze_file would then think the FOV is already fully
                                                   #   analyzed and skip writing the real one). Every other step
                                                   #   is then pure in-memory arithmetic over that one cached
                                                   #   read, no further disk I/O: (1) across every FOV and z,
                                                   #   find the single frame with the highest MEAN intensity
                                                   #   (analysis.fov.counter_mean) and use it as the reference
                                                   #   frame for threshold estimation — not a fixed z pooled
                                                   #   across FOVs (an earlier version's approach, which
                                                   #   produced a poor, non-bimodal pooled histogram once it
                                                   #   became clear some FOVs are blank at that fixed z);
                                                   #   displays that one frame's actual image (Counters have no
                                                   #   spatial information, so this needs one extra single-frame
                                                   #   re-read) plus two re-binned views of its exact histogram
                                                   #   (analysis.fov.rebin_counter) — log-scale (drives the
                                                   #   automatic threshold estimate, reusing acquisition.mosaic.
                                                   #   _estimate_bimodal_threshold's exact peak-finding) and
                                                   #   linear-scale from its minimum value to a percentile
                                                   #   cutoff (analysis.fov.counter_percentile) — review both,
                                                   #   override THRESHOLD manually if the estimate looks wrong;
                                                   #   (2) once THRESHOLD is fixed, derive every FOV's true-
                                                   #   pixel-count (NTP) profile directly from its cached
                                                   #   Counter (analysis.fov.ntp_profile_from_counters /
                                                   #   ntp_from_counter), reporting z_first_um/z_last_um (the
                                                   #   shallowest/deepest z with NTP > NTP_THRESHOLD) and
                                                   #   is_contiguous (False if signal turned off and back on
                                                   #   somewhere in between -- debris, folded tissue, noise; an
                                                   #   earlier version stopped scanning at the first z that
                                                   #   failed, assuming signal only decreases with depth, but
                                                   #   real data falsified that assumption, so every z is now
                                                   #   read/counted). Live progress/ETA via progress_display.
                                                   #   ProgressReporter while the backfill runs. Deliberately
                                                   #   NOT process-pool-parallelized: an earlier version sized a
                                                   #   pool off config.resolved_n_workers (os.cpu_count() - 2,
                                                   #   the same convention FOVScheduler uses locally) and
                                                   #   crashed with BrokenProcessPool on a shared SLURM node —
                                                   #   os.cpu_count() reports the node's total cores, not the
                                                   #   job's actual memory allocation, so the pool over-
                                                   #   subscribed and got OOM-killed; reading only CHANNEL_NM's
                                                   #   frames (not the whole multi-color stack) keeps the
                                                   #   sequential version fast without a pool. Saves the
                                                   #   reference-frame image, its histogram (log + linear
                                                   #   views), and two side-by-side heatmaps (z_first_um,
                                                   #   z_last_um, one shared color scale) to a new analysis/
                                                   #   figures/ subfolder (no prior convention existed for this)
                                                   #   plus a results CSV. No SLURM needed — only reads/writes
                                                   #   small Counter cache files unless backfilling.
                                                   #   Section 8 (experimental vs theoretical timing):
                                                   #   compares the THEORETICAL per-frame time (this round's
                                                   #   HAL <exposure_time> via acquisition.configs.
                                                   #   read_hal_exposure_time, same 0.25s fallback as
                                                   #   acquisition.dave.estimate_dave_experiment -- exposure
                                                   #   time only, no stage-move/fluidics/readout overhead)
                                                   #   against the EXPERIMENTAL per-frame time measured
                                                   #   directly from real file-write timestamps -- the
                                                   #   median gap between consecutive FOV files finishing
                                                   #   (sorted by common.io.path_mtime, zarr-aware and
                                                   #   robust to on-disk listing order), divided by the
                                                   #   round's frame count. Prints both, then feeds the
                                                   #   EXPERIMENTAL rate (which captures whatever real
                                                   #   overhead the theoretical estimate can't see) into
                                                   #   section 9. Section 9 (what-if): given a per-FOV z
                                                   #   cutoff (min(z_last_um + Z_MARGIN_UM,
                                                   #   Z_MAX_TRIMMED_UM)), estimates how much less disk
                                                   #   space/acquisition time the round would need if
                                                   #   trimmed to that depth -- reusing frame_table
                                                   #   (section 3), results_df (section 6), and
                                                   #   experimental_time_per_frame_s (section 8) directly,
                                                   #   no new image reads. Every frame_table color group
                                                   #   whose z actually varies (a real focus sweep,
                                                   #   including any blank/return-to-bead-z frames) is
                                                   #   assumed to scale with the SAME per-FOV cutoff found
                                                   #   for CHANNEL_NM; fixed (non-z-swept) frames are
                                                   #   unaffected; an FOV with no detected signal is assumed
                                                   #   to need 0 z-swept frames. N_ROUNDS_LIKE_THIS
                                                   #   extrapolates to the whole experiment -- defaults to 1
                                                   #   (this round only), NOT auto-derived from
                                                   #   meta.n_rounds, since different rounds can have
                                                   #   different color/z-sweep configurations. Saves a
                                                   #   per-FOV trim CSV.
data/
  configs/
    hal/            # hal-config-{mic}.xml — HAL config templates (one per microscope)
    kilroy/         # kilroy-config-*-{mic}-*-{YYMMDD}.xml — Kilroy configs (one or more per microscope)
    merlin/         # shared MERlin reference files, copied from R:\Software\merfish-parameters\
                    #   (2026-active files only; see prepare_imaging/07 + merlin_config.py):
      microscope/    #   MERFISH{3,4,5}.json, STORM2FUSION_2304_60xSil.json (ST2)
      codebooks/     #   C3v1_codebook.csv, LT1v0_codebook.csv, LT2v0_codebook.csv
      snakemake/     #   cluster_resource_allocation_basic.json (template create_cluster_
                    #     resource_allocation transforms)
    fishtank/       # shared fishtank reference material for lineage_tracing/lineage experiments,
                    #   copied from a real reference experiment (see fishtank_config.py):
      reference/v2/  #   intBC_codebook_v2.csv, {HEK3,EMX1,RNF2}_weights_v2.csv,
                    #     embryo_integration_whitelist.txt — dispatched by LINEAGE_LIB_VERSION
      scripts_static/ #  plot_drift.py, slurm_stats.sh, check_segmentation.ipynb — generic
                    #     utilities copied verbatim into every new experiment's fishtank/scripts/
  positions/        # boundary_positions.txt, hole*.txt — example tissue boundary files
    examples/       # ready-made boundary sets for each layout, used as the notebook-02
                    #   fallback when SAMPLE_DIR/positions is empty:
                    #   legacy/ (one boundary), single/ (1 tissue, 2 boundaries),
                    #   multi/ (2 tissues x 2 boundaries)
  readouts.csv      # default codebook readout table (bit number -> readout name), read by prepare_imaging/05
```

## Architecture

### Pre-experiment workflow

Run the eight `prepare_imaging/<variant>/` notebooks (variant = `reference`) in
order before starting the microscope. For `tumor` and `lineage_tracing` the eight
notebooks live one level deeper under an acquisition-type subfolder
(`prepare_imaging/tumor/{epi,disk}/`, `prepare_imaging/lineage_tracing/{merfish,lineage}/`);
run the set for the acquisition being prepared.

**01** (`prepare_imaging/<variant>/01_create_hal_config_and_shutters.ipynb`): defines the imaging sequence as a *frame table* (one row per camera frame, columns `color`, `channel`, `z`) using `get_frame_table`. Supports `scan_mode="interleaved"` (all colors per z-plane, AOTF) or `scan_mode="sequential"` (full z-sweep per color, boustrophedon, physical shutters). The objective's return to `bead_z` after the stack is controlled by `z_return_mode`: `"progressive"` (default) steps down with blank frames in increments of `return_step` (5 µm default); `"instant"` jumps straight back (the previous behaviour). A per-channel `POWER = {nm: power}` dict sets each shutter `<event>`'s `<power>` (actual acquisition power, by frame colour) and the HAL `<default_power>` (channel-ordered via `power_dict_to_channel_list`). Auto-generates a compact colour name via `get_color_sequence_name` (underscore-joined tokens, e.g. `blkf5_488f2_560f25_650f25_750f25`). Sets `<filetype>` (`.zarr` default, or `.dax`/`.tiff`) and `<exposure_time>` in the HAL config. Also writes a **transit** HAL config/shutter (`get_transit_frame_table`, `N_TRANSIT_BLANK` blank frames at bead z) for the between-boundary transit FOVs.

**Naming rule.** Each round's three artefacts share a stem `{kind}-{name}` (kind = `bits`/`cells`/`transit`), built by the `configs` helpers `hal_config_filename` / `shutter_filename` / `frame_table_filename`. Hyphens delimit the structural prefix; underscores live only inside `{name}`. So for the bits round with `{name}=blkf5_488f2_560f25_650f25_750f25`:
- `SAMPLE_DIR/settings/hal-config-{mic}-bits-{name}.xml` — patched from `data/configs/hal/hal-config-{mic}.xml`
- `SAMPLE_DIR/settings/shutter-bits-{name}.xml` — shutter event XML
- `SAMPLE_DIR/metadata/frame-table-bits-{name}.csv` — frame table
- `SAMPLE_DIR/metadata/shutter_sequence_{name}.png` — visualisation

(cells and transit rounds follow the same pattern with their own `kind`.) The analysis-side `find_frame_table_for_hal_config` mirrors this — it reads `<shutters>`, rewrites the `shutter-` prefix to `frame-table-`, and finds the CSV (legacy `frame_table_{name}.csv` still accepted as a fallback).

Both XML files use Windows CRLF line endings and ISO-8859-1 encoding as required by HAL.

**02 -- two independent notebooks, same input contract.** Deriving boundaries from a mosaic (`02_create_boundary_from_mosaic.ipynb`) is a self-contained analysis of `SAMPLE_DIR/data/mosaic10x/`, independent of the FOV-grid generator (`02_create_positions_from_boundaries.ipynb`) — the two used to be a single "notebook 02a feeds notebook 02" pipeline, but since a mosaic can be re-run/re-thresholded any number of times without touching FOV-grid generation (or skipped entirely for a hand-drawn boundary), they write to two separate subfolders that both feed the same downstream contract: `SAMPLE_DIR/positions/boundaries/manual/` (hand-drawn) and `SAMPLE_DIR/positions/boundaries/from_mosaic/` (notebook output). `resolve_boundaries_source_dir(positions_dir, source=None)` resolves which one to read: with `source=None` (the default in both notebooks 02 and 03) it auto-picks whichever has files, preferring `from_mosaic` — so drawing a mosaic-derived boundary automatically takes over from a manual one without any notebook edits, and notebooks 02/03 always agree without passing state between them. Both notebooks are described below.

**02 — mosaic** (`prepare_imaging/<variant>/02_create_boundary_from_mosaic.ipynb`, optional): derives `boundary_positions*.txt`/`hole*.txt` automatically from a Steve low-mag mosaic (`MERci.acquisition.mosaic`), instead of drawing them by hand. `load_steve_mosaic` reads a Steve `.msc` manifest + its `.stv` tile pickles from `SAMPLE_DIR/data/mosaic10x/` (each tile already carries its own stage position, pixel size, and `zvalue`); `assemble_mosaic_canvas` pastes **all** tiles into one flattened image in stage-micron coordinates at a single working resolution — mixed objectives/exposures and overlapping tiles (e.g. a handful of high-mag alignment FOVs shot over part of the low-mag scan) are handled directly: each tile is independently downsampled from its own real pixel size, and tiles are painted in ascending `zvalue` order so an overlapping pixel always takes the value from whichever tile is physically on top, rather than being averaged or rejected. The intensity-histogram threshold estimate (`plot_tile_intensity_histograms`) is computed from the majority-objective tiles only, so a handful of differently-exposed tiles don't add a spurious third mode and bias the estimate; `segment_mosaic_tissue` smooths, thresholds (seeded from that estimate, or Otsu), and morphologically cleans up the full composited canvas (closing bridges small real gaps, opening drops noise specks, an outward dilation adds a hand-drawing-like safety margin) before tracing tissue/hole polygons — a single global threshold on the raw canvas is not enough on its own: illumination vignetting and tile seams otherwise fragment one tissue mass into hundreds of tiny disjoint specks. A detected hole can itself contain a real island of tissue (a donut/annulus shape) rather than being a simple region to exclude entirely; `segment_mosaic_tissue` detects this per labeled component and represents it as an interior ring of the hole's polygon (`min_island_area_um2` controls the minimum size to keep as a real island vs. noise), and `save_boundary_from_mosaic` writes each island as a `hole{n}_island{m}.txt` companion file alongside the hole's own `hole{n}.txt`. This is a visual, iterative notebook: `plot_mosaic_segmentation` overlays the detected tissue (green) / hole (red, with island interiors dashed) polygons on the canvas so threshold/morphology parameters can be re-tuned and re-run before committing; only then does `save_boundary_from_mosaic` write the files into `SAMPLE_DIR/positions/boundaries/from_mosaic/`, in the exact convention `discover_boundary_files`/`load_hole_polygons` already expect (legacy `boundary_positions.txt` for one detected piece, `boundary_positions_{b}.txt` for several disjoint pieces — holes stay global, same as everywhere else in the pipeline). Validated against a real Steve mosaic + a manually-drawn ground truth: precise IoU against the hand-drawn polygon is not the goal (a hand-drawn boundary is a coarser, more generous envelope than a signal-following threshold trace), but the detected outline visibly follows the real tissue shape and finds the same internal holes, including the one real island present in the test mosaic.

**02 — FOV grid** (`prepare_imaging/<variant>/02_create_positions_from_boundaries.ipynb`): builds the FOV scanning positions for one or more tissue sections. Reads boundary files via `resolve_boundaries_source_dir` (see above). If neither `positions/boundaries/manual/` nor `positions/boundaries/from_mosaic/` has files yet, it falls back to a bundled example dataset under `MERci/data/positions/examples/{legacy,single,multi}` (chosen by the notebook's `EXAMPLE_LAYOUT`; per-variant default: tumor→`legacy`, lineage_tracing/reference→`multi`) via `resolve_boundary_dir`, and **copies that example's boundary + hole inputs into the resolved boundaries source folder** so the experiment folder is self-contained and notebook 03 (which reads the same resolved folder) finds them. This lets the whole pipeline be run and tested before any real boundaries are drawn; the copy is idempotent (skipped once the folder has inputs). It **auto-detects tissue count** from the boundary filenames in the resolved directory (`discover_boundary_files`): `tissue_{t}_boundary_positions_{b}.txt` → **multi** (several sections), `boundary_positions_{b}.txt` → **single** (one section, several boundaries), or a lone `boundary_positions.txt` → **legacy** (one boundary). Independent of that tissue count, **each tissue's own boundaries** (if it has more than one) connect to each other per `TISSUE_PATH_MODE`/`TISSUE_PATH_MODE_OVERRIDES` (a dict keyed by tissue index), defaulting to `"legacy"` for every tissue: `"legacy"` concatenates that tissue's boundaries directly into one continuous path (each boundary built with `build_boundary_path(..., return_side="top")` to close its own loop, no dedicated transit FOVs) — appropriate when a tissue's boundaries are close/contiguous; `"transit"` bridges each consecutive pair with a dedicated **transit** segment (`create_transit_path`: FOVs on the A→B line spaced ~`TRANSIT_SPACING`×step), the original multi-boundary behaviour. A tissue whose own boundaries are far apart in `"legacy"` mode (gap > 5× the grid step) gets a printed warning suggesting a `TISSUE_PATH_MODE_OVERRIDES` entry, since nothing bridges that gap. Boundaries belonging to **different tissues** always get a transit bridge regardless of this setting (physically separate coverslip regions), and the whole final segment sequence wraps its last segment back to its first so one Dave round returns near where it started. `hole*.txt` polygons are global (applied to every boundary). Writes per-segment files referenced by Dave (`positions_{SAMPLE_NAME}_{T#B#|B#|T#}.txt`, or no per-segment file for a tissue collapsed into one `"legacy"` segment) and `positions_{SAMPLE_NAME}_transit_{k}.txt`, per-tissue FOV-only files (`positions_{SAMPLE_NAME}_T{t}.txt`, or `positions_{SAMPLE_NAME}.txt` for single/legacy — skipped if a `"legacy"`-collapsed segment already wrote the identical file), and creates the `data/` subfolders for the layout (`mosaic10x`, and `tissue_{t}/{cells,hybs,transit}` or top-level `{cells,hybs,transit}`).

FOV grid rules (`create_grid_positions(..., direction=...)`): one axis is forced ODD (a cell exactly at the boundary's bounding-box midpoint), the other EVEN, chosen from `direction` — columns even for `direction="vertical"`, rows even for `"horizontal"` — so `generate_scanning_path`'s boustrophedon snake (called with the same `direction`) starts and ends in the same row/column (a short return leg) instead of the opposite corner (which an all-odd grid produces, since the first and last traversal-axis index then share the same parity/sub-direction). A FOV is kept if its camera square overlaps the boundary polygon at all; excluded only if a hole polygon fully contains the FOV square.

**03** (`prepare_imaging/<variant>/03_create_round_info.ipynb`): generates `round_info.csv`. Resolves the same boundary source folder as notebook 02 (`resolve_boundaries_source_dir(POSITIONS_DIR)`, `BOUNDARY_SOURCE = None` auto-picks whichever has files) before calling `discover_boundary_files`, so the two notebooks can never disagree about which boundary set (manual vs. from_mosaic) is in use. It also has its own `TISSUE_PATH_MODE`/`TISSUE_PATH_MODE_OVERRIDES` (same shape as notebook 02's, defaulting to `"legacy"`) that **must match whatever notebook 02 was actually run with** — both call `MERci.acquisition.positions.group_boundaries_by_path_mode(boundaries, MODE, tissue_path_mode)` (the single source of truth for which tissue's boundaries got merged), and notebook 03 picks its recipe from the resulting group count rather than the raw boundary count: exactly one group (e.g. a tissue's boundaries all merged under `"legacy"`) uses the classic single-positions recipe (`create_round_info`); **more than one group** builds a **segment-aware** `round_info` (`create_round_info_multitissue`, itself calling the same `group_boundaries_by_path_mode` and taking a `tissue_path_mode` argument: one row per (round, segment) — boundary movies with the cells/bits config + transit movies with the transit config, plus `positions_file`, `tissue`, `segment` columns; a merged/legacy segment's `positions_file` is the plain aggregate `positions_{sample}.txt`, matching notebook 02's own convention). HAL configs for bits vs. cells rounds are auto-detected by glob patterns (`blkf3*` for bits, `blkf1*` for cells); the transit HAL config from notebook 01 is auto-detected too. This notebook also **defines the round–bit–colour mapping** (`round_bit_color`, one `(round, bit, color_nm)` per bit) and **derives `N_HYBS` from it** (`N_HYBS = max(round)`) rather than hard-coding it, so the hyb count always matches the codebook; it saves the mapping to `SAMPLE_DIR/metadata/round_bit_color_map.csv` for notebooks 04-07 to reuse. Writes `SAMPLE_DIR/metadata/round_info.csv` and `SAMPLE_DIR/metadata/round_bit_color_map.csv`.

**Multi-drive round-robin (optional).** Setting `DATA_DRIVES = ["D:", "E:", "F:"]` (default `[]` = disabled) makes `create_round_info`/`create_round_info_multitissue` spread successive **hyb** rounds round-robin across those drives — round *i*'s `data_dir` becomes `<drive>/data/hybs/H{NN}` instead of always `SAMPLE_DIR/data/hybs/H{NN}` (cells/transit are unaffected, always under `SAMPLE_DIR/data/...`). `create_data_drive_skeleton` pre-creates each hyb's `H##` folder only on the one drive it is actually assigned to (via the same round-robin assignment as `round_info.csv`, not on every configured drive) — so which hybs live on which disk is visible directly from each drive's folder listing, and can never disagree with `round_info.csv`. This pairs with `analysis_mode="round_robin_drives"` (see Online-analysis architecture below) so analysis can read already-completed rounds on idle drives while HAL is still writing the current round to a different one.

**04** (`prepare_imaging/<variant>/04_create_dave_config.ipynb`): builds the Dave experiment recipe XML from `round_info.csv` (notebook 03). With a single boundary it uses the classic single-positions recipe (one `<loop>` per imaging round). With **multiple boundaries** it builds a **per-segment** Dave recipe (`create_dave_config(positions_dir=…)`: each boundary/transit segment is its own `<loop>` — "Imaging Round NN - <segment>" — with its own movie, HAL config and positions file, in order; fluidics loops stay between rounds). Every `<loop>` gets its own identically-named `<loop_variable>`, even when several rounds share one positions file: Dave's real `v2Generator` (`handleLoop`) resolves a loop strictly by `self.loop_variable_names.index(loop.attrib["name"])`, so a movie's `<variable_entry>` cannot reference a differently-named loop_variable declared elsewhere — a shared/deduped loop_variable name loads fine in MERci's own reader but makes real Dave raise `ValueError: '<loop name>' is not in list` (confirmed directly against the storm_control source). The Kilroy config for the microscope is resolved (via `find_kilroy_config`, falling back to MF2 when the microscope has no config) and passed to `create_dave_config` as the source of fluidic protocol names: every protocol written into the Dave recipe is resolved to — and required to exist as — a `<protocol>` in that Kilroy config, raising `ValueError` otherwise. Writes `SAMPLE_DIR/settings/dave-{mic}-{N}hybs-{SAMPLE_NAME}.xml`, named via `dave.dave_config_filename(microscope, n_hybs, sample_name)` — a single source of truth for this filename shared with notebook 05's annotation step below.

**05** (`prepare_imaging/<variant>/05_create_data_organization.ipynb`): generates the MERlin data-organization CSV and annotates the Dave XML (notebook 04) with per-round bit information. Picks the bits/cells series by `imaging_type` (so a multi-boundary `round_info`'s transit movies are never selected). Note: multi-tissue MERlin analysis is per tissue / per boundary — confirm the intended workflow before relying on the generated data-organization. Requires `MERci/data/readouts.csv` (codebook mapping bit numbers to readout names; shipped in the repo). Frame tables and series patterns are auto-detected from `metadata/`. The `round_bit_color` mapping is **defined in notebook 03**; this notebook reads it back from `SAMPLE_DIR/metadata/round_bit_color_map.csv` (raising `FileNotFoundError` if notebook 03 has not run). The Dave file to annotate is located via the same `dave.dave_config_filename(microscope, n_hybs, sample_name)` notebook 04 used to write it — not by globbing `settings/dave-*.xml` and taking the alphabetically-last match, which silently annotated the wrong file whenever two acquisitions with different hyb counts shared one `settings/` folder (e.g. a 13-hyb and a 9-hyb experiment: `"...-13hybs-..."` sorts *before* `"...-9hybs-..."` as a string, so `sorted(...)[-1]` picked the 9-hyb file). Writes:
- `SAMPLE_DIR/metadata/data_organization_{MICROSCOPE}_{SAMPLE_NAME}.csv`
- Annotates `SAMPLE_DIR/settings/dave-{mic}-{N}hybs-{SAMPLE_NAME}.xml` with per-round bit comments

**06** (`prepare_imaging/<variant>/06_create_experiment_info.ipynb`): writes `SAMPLE_DIR/metadata/experiment_info.yaml` — a small, human-readable per-experiment record mirroring the master per-project experiment-info CSVs kept outside the repo (e.g. `experiment_info/lt_experiment_info.csv`), so many experiments' files can later be batch-collected back into one of those tables via `experiment_info.collect_experiment_info`. Auto-fills what MERci already knows (bit count from `round_bit_color_map.csv`, exposure time read from a bits HAL config via `configs.read_hal_exposure_time`, positions file(s) present in `positions/` — matched against `SAMPLE_NAME`, the same true id notebooks 02-05 already use to name their files — and `acquisition_type` — `"epi"` or `"disk"` — derived from `MICROSCOPE` via `configs.get_acquisition_type`). `SAMPLE_NAME`/`IMAGING_DIR` (the true top-level experiment id and acquisition-type subfolder) are auto-detected from the folder structure via `experiment_info.resolve_sample_identity`, not typed by hand: once an acquisition lives under its own subfolder (e.g. `SAMPLE_DIR = .../LT058_sample_07/merfish`), `SAMPLE_DIR.name` is just this acquisition's local tag, not the true id one level up — getting this wrong previously made both the local positions-file naming (`positions_merfish.txt` instead of `positions_LT058_sample_07.txt`) and the cluster-facing `DATA_HOME`/`MERLIN_HOME`/`FOLDER_NAME` fields below double up (e.g. `"merfish/merfish/data"`). Leaves those cluster destination paths and biology/sample metadata (fix type, hyb temperature, tissue type, …) as a parameters cell for the user, since MERci has no way to know them.

**07** (`prepare_imaging/<variant>/07_create_merlin_scripts.ipynb`): generates the remaining MERlin input/run files into `SAMPLE_DIR/merlin/`, reading `experiment_info.yaml` (notebook 06). Resolves the codebook/microscope-parameters files shipped in `MERci/data/configs/merlin/{codebooks,microscope}/` by `lib_name`/`microscope` (`merlin_config.resolve_codebook_filename`/`resolve_microscope_parameters_filename`) and sanity-checks the codebook's own bit count against `round_bit_color_map.csv` before proceeding. Builds the analysis-parameters JSON from a `MerlinAnalysisSpec` (`create_merlin_analysis_parameters` — which steps to include: `n_optimize_iterations`, `include_reporting`, `include_segmentation` + method) rather than copying and hand-editing a prior experiment's file, then the cluster-resource-allocation JSON, snakemake parameters JSON, and the slurm submit script (`merlin_config.create_*`) — all named using `SAMPLE_NAME` (`info.sample_name`, the true experiment id from `experiment_info.yaml`), consistent with the data-organization CSV and positions file (`metadata/`/`positions/`, notebooks 05/02) they reference by their existing paths. The same `SAMPLE_NAME` is also passed to `resolve_cluster_sample_dir`'s project-root/cluster-path resolution — the shared codebook/microscope files are referenced by their path inside this `MERci/` clone directly, since the clone already lives inside `SAMPLE_DIR/`, no file needs to be copied to a separate shared cluster location.

**`lineage_tracing/lineage` only — 05/07 are fishtank-based, not MERlin.** This
acquisition type is analyzed with **fishtank**, so its notebooks 05 and 07
differ from every other variant:

**05** (`lineage_tracing/lineage/05_create_color_usage.ipynb`): writes
fishtank's `color_usage` CSV (`fishtank_config.create_color_usage_csv`) — a
per-round/color target table whose round-tag mapping (`"r1"`, `"beads"`,
`"DAPI"`, `"empty"`, …) is entered **manually**, since it's a per-protocol
choice not derivable from `round_info.csv`/`round_bit_color_map.csv`. Also
writes a second, single-row `color_usage_{SAMPLE_NAME}_mf.csv` naming the
sibling merfish acquisition's cells/DAPI round (for cross-modality
registration) and a `decoding_strategy` CSV
(`create_decoding_strategy_csv` — per-target decode method + reference file).
Writes `SAMPLE_DIR/metadata/color_usage_{SAMPLE_NAME}{,_mf}.csv` and
`SAMPLE_DIR/metadata/decoding_strategy_{SAMPLE_NAME}.csv`.

**07** (`lineage_tracing/lineage/07_create_fishtank_scripts.ipynb`): builds
`SAMPLE_DIR/fishtank/` — the folder skeleton (`create_fishtank_folder_skeleton`:
`params/`, `reference/`, `scripts/`, `output/`, `log/`, plus the shared
static utility scripts), the shared reference files for the configured
`LINEAGE_LIB_VERSION` (`copy_fishtank_reference_files`, dispatched via
`resolve_fishtank_reference_dir` — mirrors the MERlin codebook dispatch), and
every fishtank run script from a `FishtankScriptsSpec` (`create_fishtank_scripts`
— cellpose segmentation for both acquisitions, spot detection/decoding,
DAPI mosaics for cross-modality registration; every field overridable,
defaulting to a verified reference experiment's values). Assumes the
confirmed sample layout — the lineage and merfish acquisitions of one sample
are sibling directories (`<sample_id>/{merfish,lineage}/`, `fishtank/` inside
the lineage one) — so `MERFISH_SAMPLE_DIR` defaults to `SAMPLE_DIR.parent /
"merfish"`, and the merfish acquisition's positions file is copied into the
lineage acquisition's own `positions/` folder (fishtank's generated scripts
resolve both positions files from one folder, `../../positions/` relative to
`fishtank/scripts/`). Both sibling acquisitions now name their own positions
file with the same true `SAMPLE_NAME` (notebook 02), so the copied-in merfish
file is renamed with a `_merfish` suffix on copy to avoid overwriting the
lineage acquisition's own positions file of the same name.

### Online-analysis architecture

`ExperimentConfig` holds all paths and tunable parameters. Notable fields:
- `image_suffix` — `.zarr` (default), `.dax`, or `.tiff`
- `fluidics_type` — `"adaptor"` (t_max = 100 min) or `"direct"` (t_max = 50 min); sets `t_max` automatically when left as `None`
- `settings_dir` — `SAMPLE_DIR/settings/`; needed for auto flip_y and per-color mosaic lookup
- `mosaic_flip_y` — `None` (auto-read from HAL config `<flip_vertical>`), `True`, or `False`
- `fov_subset` — list of FOV ids to restrict analysis; `None` = all FOVs
- `transfer_dest` — network path (e.g. a NAS) to copy completed round data to; `None` = disabled. In `same_drive`/`mirror_drive` mode this only runs during the fluidics window (see `transfer_min_time`); in `round_robin_drives` mode it runs **continuously** via `RoundScheduler._process_pending_transfers`, skipping only the round on the drive HAL is actively writing, but still gated on that round's mosaics already being built locally (`tracker.is_round_done`) — since each round already lives on its own physical drive, a completed round elsewhere is always safe to copy out. See `TransferScheduler` (below) for the alternative that decouples transfer from local analysis entirely, for projects moving QC to a cluster.
- `transfer_min_time` — minimum seconds remaining in the fluidics window before starting a transfer (`same_drive`/`mirror_drive` modes only)
- `analysis_mode` — `"same_drive"` (default, mode B: analyse from `data_dir`), `"mirror_drive"` (mode A: mirror `data_dir` → `analysis_source_dir` during fluidics and analyse from that second-drive copy), or `"round_robin_drives"` (for experiments whose `round_info.csv` spreads hyb rounds across several physical drives via `create_round_info(data_drives=…)` — see prepare_imaging/03). Analysis runs **continuously** in all three modes (not only during fluidics). `config.analysis_data_dir` resolves to the directory the FOV scheduler reads from (not meaningful in `round_robin_drives` mode — see `all_data_roots` below).
- `analysis_source_dir` — second-drive mirror directory; **required** when `analysis_mode="mirror_drive"`
- `all_data_roots` — every directory referenced in `round_info_csv`'s `data_dir` column plus `data_dir` itself, read directly from the CSV; used by `FOVScheduler`/`ExperimentStateMonitor` in `round_robin_drives` mode to scan every physical drive instead of one root. No new config field is needed for the drive list itself — `round_info.csv` is the single source of truth.
- `n_analysis_workers` — FOV process-pool size; `None` = `cpu_count − 2` (`config.resolved_n_workers`). Each worker holds one image stack (~200 MB) in RAM.

`ExperimentMetadata` (loaded via `ExperimentMetadata.load(round_info_csv, positions_txt, data_dir)`) cross-references round IDs, FOV IDs, series patterns, and expected file paths. When a `dir`/`data_dir` column is present in `round_info.csv`, per-round file paths are resolved from that directory instead of the top-level `data_dir` — and that column's value can be **either** an absolute path (round-robin rounds on the microscope, genuinely on their own physical drive, e.g. `D:/.../data/hybs/H01`) **or** a path relative to *this machine's own* `data_dir`'s parent (e.g. `data/hybs/H01` — what a transferred experiment's round_info.csv copy carries after `transfer.rewrite_round_info_dirs`, since the original absolute drive letter is meaningless once consolidated onto a NAS/cluster). Each series carries an ordered list of **candidate directories** (`SeriesInfo.candidate_dirs`); `resolve_path(fov, suffix)` returns the first candidate that exists on disk. If none of the exact-width-guessed candidates exist, it falls back to a **width-agnostic directory scan** (`SeriesInfo._scan_dir_for_fov`, cached per directory so 1000+ per-FOV lookups against one round only pay for one scan): `fov_from_stem` was already built from `\d+` (never a fixed digit count — see `_pattern_to_regex`), just previously only used in the reverse (path → FOV id) direction; this reuses it forward too, so a `round_info.csv` whose `series` pattern's zero-pad width has gone stale (e.g. positions.txt regenerated with more FOVs, needing more digits, after round_info.csv was already written) still finds the real files instead of silently reporting them all missing. Only once that scan also comes up empty does it fall back to the primary candidate (e.g. before acquisition, when the FOV genuinely doesn't exist yet). The **cells round** is treated as a bona fide imaging round (typically `imaging_round=1`) and its files are accepted in **either** `data/cells/` or the top-level `data/`, regardless of which the `data_dir` column records — so `all_fovs_done_for_round`, mosaics, and transfers all find the cells data wherever HAL actually wrote it. Two more methods support `round_robin_drives` mode: `drive_of_round(round_id)` returns the drive letter of a round's `data_dir`, and `actively_writing_round()` returns the round HAL is most likely writing right now — the highest round id that has started (≥1 expected file exists) but isn't yet complete (not every expected file exists); it returns `None` once that round is fully on disk, so a just-finished round's drive isn't wrongly excluded for its whole following fluidics window.

`ExperimentStateMonitor` determines the microscope phase by watching the newest file mtime in `data_dir` (or, in `round_robin_drives` mode, across every drive in `config.all_data_roots`):
- **IMAGING**: a new image file was written within `imaging_idle_threshold` seconds
- **FLUIDICS**: `t_min ≤ time_since_imaging ≤ t_max` → `should_analyze = True`

`should_analyze` is no longer the analysis gate — FOV/round analysis runs continuously. The phase is still used to time the mode-A mirror and the NAS transfer (both read the acquisition drive, so both run only while the microscope is idle).

`ProgressTracker` tracks completeness via zero-byte sentinel files under `analysis_dir/done/`:
- `<stem>.fov_done` — FOV analysis complete
- `round_<r>.round_done` — mosaic(s) built for round r
- `round_<r>.round_transferred` — raw data for round r copied to `transfer_dest`
- `round_<r>.fov_submitted` / `round_<r>.round_mosaic_submitted` — cluster-side SLURM submission bookkeeping (hold the submitted job id as text, not just an empty touch — see "Moving QC analysis to a SLURM cluster" below)

Multiple notebooks can run concurrently — no shared state.

`FOVScheduler.run_loop()` runs **continuously** (acquisition + fluidics): each tick it (in mirror mode) refreshes the second-drive mirror while idle, discovers stable image files (zarr/dax/tiff) under `config.analysis_data_dir` (or every drive in `config.all_data_roots` in `round_robin_drives` mode — each root's scan is wrapped so an unreachable/disconnected disk logs a warning instead of crashing the tick), and analyses pending files **in parallel across a process pool** (`config.resolved_n_workers`). In `round_robin_drives` mode, files on the drive of `meta.actively_writing_round()` are excluded from `pending` each tick — HAL's current write target is skipped, every other drive's completed rounds analyse immediately. Each worker runs the top-level `analysis.fov.analyze_file`, which reads the stack once and writes thumbnails (PNG) + per-frame stats (CSV) + histograms (`.npz`) + the FOV sentinel. With `n_analysis_workers=1` it runs serially in-process. Respects `fov_subset`; call `.close()` (done automatically when `run_loop` exits) to shut the pool down. `RoundScheduler.run_loop()` also runs continuously, assembling **one mosaic per imaging color** (`round_{r:03d}_{color}nm_mosaic.png`) once all FOV sentinels exist; auto-resolves `flip_y` from the HAL config; optional background transfers via `transfer.transfer_round` happen only during fluidics in `same_drive`/`mirror_drive` mode, or continuously (same active-round/drive exclusion as `FOVScheduler`) in `round_robin_drives` mode. `ExperimentScheduler.wait_and_run()` calls a user callback after all rounds complete. `FOVScheduler._build_task`/`RoundScheduler._analyse_one_round`'s path/kwarg-construction logic is factored out into module-level `build_fov_task_kwargs`/`build_round_mosaics`/`resolve_round_flip_y`/`resolve_round_color_frame_indices`/`source_dirs_for_round` functions in `scheduler.py` — both the schedulers and the cluster-side CLI scripts below call these, so local and cluster analysis can never disagree about where outputs land.

**Moving QC analysis to a SLURM cluster (`round_robin_drives` projects).** `01_fov_scheduler.ipynb`/`02_round_scheduler.ipynb` remain fully supported for `same_drive`/`mirror_drive` projects, but for `round_robin_drives` projects QC analysis can instead run entirely on a SLURM cluster, freeing the microscope computer:

- **`06_transfer_to_nas.ipynb`** (microscope computer) — `TransferScheduler` (in `scheduler.py`) transfers each round's raw data to `config.transfer_dest` as soon as it's **fully written** (`ExperimentMetadata.round_fully_written(round_id)`: every expected raw file exists — no dependency on any local analysis sentinel), gated only on not being on the drive `meta.actively_writing_round()` is currently hot on — unlike `RoundScheduler`'s `round_robin_drives` transfer path, which additionally requires `tracker.is_round_done` (mosaics already built locally) and uses the simpler, unverified whole-directory `transfer.transfer_round`. `TransferScheduler` instead transfers **one FOV at a time** (`transfer.transfer_fov`: every file sharing that FOV's stem, bit-for-bit SHA-256 verified against the source before being marked `.fov_transferred`) — a round becomes `round_transferred` only once every one of its FOVs has verified, and only then, if constructed with `delete_source_after_verify=True` (off by default — irreversible), is that round's entire source data directory deleted. `TRANSFER_DEST` is derived from just a destination drive via `dave.rebase_on_drive` (mirrors `SAMPLE_DIR`'s own subpath, same convention round-robin hyb rounds use) rather than typed out by hand. The notebook calls `scheduler.sync_static_metadata()` **once** (not from the tick loop) to sync `round_info.csv`/`round_bit_color_map.csv`/`positions_*.txt`/`settings/*.xml` — none of these change once acquisition starts, so re-syncing them every tick was pure overhead — alongside a one-time mirror of the equally-static `data/mosaic10x`/`MERci`/`merlin`/`fishtank` folders (`transfer.mirror_dir_sync`), so `TRANSFER_DEST` ends up a genuinely complete copy of `SAMPLE_DIR`.
- **`07_cluster_submit_analysis.ipynb`** (run on a cluster login/transfer node, after you've moved data from the NAS to cluster storage yourself — e.g. via Globus/FileZilla; no NAS→cluster automation exists in this repo) — builds `ExperimentConfig`/`ExperimentMetadata`/`ProgressTracker` pointed at the cluster-side `SAMPLE_DIR` exactly like `05_batch_sample_review.ipynb` does, then for each round with pending FOVs (`tracker.pending_fov_files`, which only ever returns files that already exist — partial/incremental arrival is fine) writes a manifest and submits a SLURM **array job** (one task per FOV) via `acquisition/cluster_submit.py`, and similarly submits a mosaic-building job once a round's FOVs are all done (`tracker.pending_rounds`). Submission is tracked with new `ProgressTracker` sentinels — `round_<r>.fov_submitted` / `round_<r>.round_mosaic_submitted` (holding the submitted SLURM job id as text, not just an empty touch) — checked against `cluster_submit.is_job_active` (an `sacct`-based query) before resubmitting, so re-running the notebook while a previous array job is still `PENDING`/`RUNNING` is a no-op.
- **`analysis/cli_analyze_fov.py` / `analysis/cli_build_round_mosaic.py`** — standalone scripts (not part of the public MERci import surface) that are the actual SLURM-array-task payload: each reads its own `sys.path` bootstrap from `Path(__file__).resolve().parents[2]` (mirroring every notebook's own `MERCI_DIR`/`sys.path.insert` dance), so **no `pip install` is needed on the cluster either** — `MERci/` just needs to exist as a repo clone alongside the data (`git clone` it directly on the cluster, or let it ride along with your NAS→cluster transfer), and the cluster conda env needs only the same scientific dependencies as `environment.yml`, mirroring `merci_env`. `cli_analyze_fov.py` reads one manifest line (an image path) at `$SLURM_ARRAY_TASK_ID` and calls `analyze_file` via `build_fov_task_kwargs`; `cli_build_round_mosaic.py` takes `--round-id` (or a manifest + array index for several rounds at once) and calls `build_round_mosaics`.
- **`acquisition/cluster_submit.py`** — sbatch script generation + submission, following `acquisition/fishtank_config.py`'s `_sbatch_header` conventions (this is FOV-parallel array work like fishtank's cellpose/detect-spots jobs, not MERlin's single-orchestrator-job convention — see `acquisition/merlin_config.py`): `build_fov_array_script`/`build_round_mosaic_script` write the sbatch text (default partition `"zhuang,sapphire,shared"`, `module load python` + `source activate merci_env`, real `#SBATCH --array=0-{n-1}%{concurrency}`), invoking the CLI scripts above by their absolute path under the cluster's own `MERci/` clone (never `python -m MERci...`, since MERci is never installed as a package). `submit_sbatch`/`job_state`/`is_job_active` wrap `sbatch`/`sacct` (the latter the same tool `data/configs/fishtank/scripts_static/slurm_stats.sh` already uses for job-resource auditing) via `subprocess`, logging and returning `None` rather than raising on any submission hiccup so a polling notebook loop never crashes.

Typical scheduler setup in a notebook:

```python
from MERci.common.config   import ExperimentConfig
from MERci.common.metadata import ExperimentMetadata
from MERci.progress        import ProgressTracker
from MERci.state           import ExperimentStateMonitor
from MERci.scheduler       import FOVScheduler, RoundScheduler

config = ExperimentConfig(
    data_dir       = SAMPLE_DIR / "data",
    metadata_dir   = SAMPLE_DIR / "metadata",
    analysis_dir   = SAMPLE_DIR / "analysis",
    settings_dir   = SAMPLE_DIR / "settings",
    round_info_csv = SAMPLE_DIR / "metadata" / "round_info.csv",
    positions_txt  = SAMPLE_DIR / "positions" / f"positions_{SAMPLE_NAME}.txt",
    image_suffix   = ".zarr",          # or ".dax" / ".tiff"
    fluidics_type  = "adaptor",        # sets t_max = 100 min
)
meta    = ExperimentMetadata.load(config.round_info_csv, config.positions_txt, config.data_dir,
                                   image_suffix=config.image_suffix)
tracker = ProgressTracker(config.analysis_dir)
monitor = ExperimentStateMonitor(config)

FOVScheduler(config, meta, tracker, monitor).run_loop()
```

### Key data files

- `round_info.csv` — required columns: `imaging_round` (or legacy `round_id`), `series` (Python format string like `hal-mf3_{fov:03d}_01`); optional: `imaging_type`, `hal_config`, `shutter_file`, `dir`. Loaded via `common.io.load_round_info`.
- `positions_{SAMPLE_NAME}.txt` — comma-separated `x,y` per line, one FOV per line; `#`-prefixed lines ignored
- Image files — HAL writes `.zarr` (directory store, default), `.dax` (raw uint16 binary + `.inf` sidecar), or `.tiff` (multi-page). Use `read_image(path)` to load any format. `discover_image_files` handles both flat files and zarr directory stores.
- HAL config templates — `data/configs/hal/hal-config-{mic}.xml`; auto-detected in `prepare_imaging/01` by microscope name; patched by `create_hal_config` (sets frames, shutters, z_offsets, filetype, exposure_time)
- Kilroy config files — `data/configs/kilroy/kilroy-config-*-{mic}-*-{YYMMDD}.xml`; resolved in `prepare_imaging/04` by `find_kilroy_config` (newest by YYMMDD; falls back to MF2 when the microscope has no config), copied to `SAMPLE_DIR/settings/`, and used as the source of fluidic protocol names for the Dave recipe. Protocol names are **not** standardised across microscopes (e.g. `Cleave Adaptors` vs `Cleave Adaptor`), so `KilroyProtocolResolver` token-matches each logical dave step (cleave / hybridize k / readouts / image buffer) to the real `<protocol>` name in the chosen config.

### Microscope channel mapping

`MF2`, `MF3`, `MF4`, and `MF5` share the same 5-channel mapping: `{405→4, 488→3, 560→2, 650→1, 750→0}`. `MFX` and `ST2` have only 4 channels with a distinct ordering: `{650→0, 560→1, 488→2, 405→3}` (no 750). `NaN` = blank frame (no laser). Extend `_COLOUR_TO_CHANNEL` in `acquisition/configs.py` for other microscopes.

Camera geometry also follows from the microscope: `MFX` and `ST2` have 2304×2304 sensors at 0.0878 µm/pixel; the MF-series (`MF2`–`MF5`) have 2048×2048 at 0.108 µm/pixel. `acquisition/configs.py` exposes `get_camera_frame_size(microscope)` (sensor pixels; mapping `_CAMERA_PIXELS`), `get_camera_pixel_size_um(microscope)` (mapping `_CAMERA_PIXEL_SIZE_UM`), and `get_fov_geometry(microscope) -> (pixel_size_um, image_size_px)` which bundles both. Frame size drives the storage figure in the Dave experiment estimate (`estimate_dave_experiment` / the summary printed by `create_dave_config`); `get_fov_geometry` gives `prepare_imaging/02` its scanning-grid geometry from the microscope alone (set `MICROSCOPE` there instead of hard-coding `pixel_size_um`/`image_size_px`).

Acquisition type (imaging modality) is a separate, orthogonal property of the microscope — independent of the channel-mapping/camera-geometry groupings above. `MF2`, `MFX`, and `ST2` are spinning-disk confocal (`"disk"`); `MF3`, `MF4`, and `MF5` are epifluorescence (`"epi"`). `acquisition/configs.py` exposes `get_acquisition_type(microscope)` (mapping `_ACQUISITION_TYPE`), used by `prepare_imaging/05` to auto-fill the `acquisition_type` field of `experiment_info.yaml`.

## Notebook coding guidelines

Every notebook -- new or edited -- follows the architecture rules in
[`NOTEBOOK_GUIDELINES.md`](NOTEBOOK_GUIDELINES.md) (repo root): separate
calculation cells from display/plot cells, cache calculation results under
`analysis/cache/<notebook_name>/`, skip recomputation when a valid cache
already exists, report progress (n/total, elapsed, ETA) in every nontrivial
calculation loop, and use explicit, legible plot font sizes. See
`notebooks/misc/measure_tissue_thickness.ipynb` for the reference
implementation.

## Running notebooks

Notebooks auto-detect `SAMPLE_DIR` from their own location. `analysis/` and `misc/` notebooks are two levels under the repo root, so `MERCI_DIR = Path(os.getcwd()).parent.parent` (the `MERci/` clone), then `SAMPLE_DIR = MERCI_DIR.parent`. The `prepare_imaging/<variant>/` notebooks (`reference`) are **three** levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent`; the `tumor/{epi,disk}/` and `lineage_tracing/{merfish,lineage}/` notebooks are **four** levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`. Do not hardcode absolute paths in notebooks.

## Scope constraint

All edits and analysis must stay within this repo. Do not modify sibling folders (`image_acquisition/`, `imaging_with_storm_control/`, etc.) unless explicitly requested.

## Version control

Commit and push as you go — do not leave finished work uncommitted.

- After completing each individual modification, create its own focused git commit
  (one logical change per commit) with a clear message, then `git push` to `origin`.
- Do not batch many unrelated changes into a single commit, and do not let edits
  pile up locally — the remote (`github.com/leonardosepulveda/MERci`) should reflect
  progress as it happens.
- This is standing authorization to commit and push without asking each time.
- Never commit transient files: atomic-write leftovers (`*.tmp.*`), `__pycache__/`,
  or `*.egg-info/` (all gitignored).

## Remembering task history

For **every user question/request**, log it to `prompt_history/` (a gitignored,
local-only folder). This lets project context be reconstructed if the
conversation history is lost. There are **two methods** depending on how the
prompt arrives:

### Method 1 — user pre-writes the prompt as a file

The user creates a file in `prompt_history/` whose name is **only the date/time**
(e.g. `2026_06_18_1002.txt`) and writes their request inside it. When asked to read
and act on it:

1. Read the file and carry out the request.
2. **Append** the response record to the *same file* (the YAML frontmatter +
   `## Plan` + `## Summary` sections below; the `## Prompt` is already the file's
   existing content). Keep the original prompt text intact — append, never rewrite.
3. **Rename** the file to add a short description, converting it to the standard
   form `{YYYY_MM_DD_HH_MM}_{short_description}.md`
   (e.g. `2026_06_18_1002.txt` → `2026_06_18_1002_update_logging_rules.md`).

### Method 2 — user types the prompt in the Claude Code window

Create a **new** file per request, named `{YYYY_MM_DD_HH_MM}_{short_description}.md`
(e.g. `2026_06_04_1432_add_prompt_history_convention.md`), with the full template
below including a verbatim `## Prompt` copy.

### Shared format

Use YAML frontmatter for queryable metadata, then prose sections. Template:

```markdown
---
date: YYYY-MM-DD HH:MM
title: <short description>
files_modified:
  - path/relative/to/repo
elapsed: <e.g. 12m 30s — wall-clock from prompt submission to completion>
status: completed | in-progress | abandoned
---

## Prompt
<verbatim copy of the user's request>

## Plan
<Claude's plan of action before executing>

## Summary
<what was actually done, including any deviations from the plan>
```

The `UserPromptSubmit` date/time hook injects `Current local date/time: … (epoch N)`
on every prompt. Compute **`elapsed`** (written just before `status`) as the finish
time minus that submit epoch: run `date +%s` (bash) or
`[DateTimeOffset]::Now.ToUnixTimeSeconds()` (PowerShell) when done and subtract the
epoch from the message that began the request (the first turn's epoch for a
multi-turn request). Omit `elapsed` if no submit epoch is available — never guess it.

Format rationale: Markdown + YAML frontmatter is Claude-native, human-readable,
and lets all entries be scanned/grepped by metadata without reading every body.

### Someday / backlog

`SOMEDAY.md` (repo root, gitignored) holds work that's real but intentionally
deferred -- e.g. a risk noticed in passing that hasn't actually caused a problem
yet, or a task paused mid-way in favor of something more urgent. One dated entry
per item, newest at the top. Not a replacement for `prompt_history/` -- when an
entry is picked up, do the work as a normal logged request and delete the entry
from `SOMEDAY.md` rather than leaving it to go stale.

### Optional: rationale docs for investigation-heavy tasks

`prompt_history/`'s Summary is compressed prose with no code — good for a
quick record, not for actually learning *how* a conclusion was reached. For
tasks with genuine investigation (reverse-engineering an undocumented
format, debugging by reading unfamiliar source, iterating on an approach
that failed at first), also write `prompt_rationales/{same-basename}.html`
(gitignored, personal-only, like `prompt_history/`/`verbatim_history/`) — a
narrative walkthrough with real code snippets, dead ends, and the moment a
hypothesis got confirmed or falsified. Not needed for mechanical tasks. See
`~/.claude/commands/rationale.md` (the `/rationale` command) for the exact
template/style, and offer one proactively at the end of an investigation-
heavy task rather than waiting to be asked — the reasoning is easiest to
capture while still fresh in context.

