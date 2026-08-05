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

Open notebooks from their folders under `MERci/notebooks/` (`prepare_imaging/`, `analysis/`, `during_imaging/`, `misc/`) in the JupyterLab file browser so that `SAMPLE_DIR` is auto-detected correctly.

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
    metadata.py     # ExperimentMetadata — parses round_info.csv + positions.txt.
                    #   `_path_exists_safe` (used by SeriesInfo.resolve_path/_scan_dir_for_fov/
                    #   ExperimentMetadata.round_fully_written) treats an OS-level access error the
                    #   same as "doesn't exist" rather than letting it propagate -- round_info.csv's
                    #   `dir` column can be a stale absolute path from a DIFFERENT machine (confirmed
                    #   directly on a real experiment: `OSError: [WinError 1326] The user name or
                    #   password is incorrect` for an unreachable drive letter), which plain
                    #   `Path.exists()` does NOT swallow (only the specific "not found" errno), so it
                    #   previously crashed candidate-path resolution instead of just skipping to the
                    #   next candidate. Note this only fixes the CRASH, not necessarily the SPEED: if
                    #   the unreachable candidate's own OS-level failure is itself slow (e.g. a network
                    #   auth retry, not an instant refusal), resolving many FOVs still pays that cost
                    #   once per FOV -- confirmed impractically slow across 1000+ FOVs x many rounds on
                    #   the same real experiment; a caller in that situation is often better off
                    #   resolving paths directly relative to its own already-confirmed-reachable
                    #   SAMPLE_DIR (see e.g. `notebooks/tests/fix_mosaic_shift_missing_fovs.ipynb`)
                    #   rather than constructing a full ExperimentMetadata at all.
    io.py           # read_dax/zarr/tiff/image, parse_inf, get_dax_shape, load_round_info, load_positions,
                    # save_positions_array, discover_image_files, path_mtime (effective last-write mtime
                    # of a file OR a directory store -- max mtime of its contents, zarr-aware, since a
                    # directory's own mtime doesn't reliably update when a chunk file nested inside it is
                    # written; used by misc/measure_tissue_thickness_test.ipynb to measure real inter-FOV
                    # acquisition timing from file-write timestamps)
                    # + selective per-frame reading (only the requested frame_indices, real partial I/O
                    #   where the format allows it -- zarr fancy-indexing, dax direct byte-offset seeking,
                    #   tifffile's own key= selection -- rather than reading the whole stack and discarding
                    #   most of it): read_dax_frames, read_zarr_frames, read_tiff_frames, read_image_frames
                    #   (format-agnostic dispatcher, mirrors read_image) -- each reads every requested frame
                    #   eagerly, built on a lazy iter_dax_frames/iter_zarr_frames/iter_tiff_frames/
                    #   iter_image_frames counterpart that yields (frame_idx, frame) one at a time, so a
                    #   caller that might stop partway through frame_indices (e.g. a sequential scan with a
                    #   per-frame stopping condition -- see analysis.fov.measure_tissue_tpc_profile) never
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
                    # get_color_to_channel_dict, create_shutter_file (every event's <power> is a
                    #   fixed default_power, e.g. 1.000, regardless of colour -- NOT per-colour;
                    #   an earlier version wrongly varied it by colour, see notebook-01 prose below),
                    # create_hal_config, format_z_offsets_from_frame_table,
                    # power_dict_to_channel_list (the ONLY place per-colour power applies: colour->power,
                    #   channel-ordered HAL default_power),
                    # find_mosaic_helper_configs / copy_mosaic_helper_configs (hand-crafted per-microscope
                    #   10x/60x mosaic-tool setup configs, kept in data/configs/hal/mosaic_helper/ -- a
                    #   dedicated subfolder so they're never picked up by the HAL-template auto-detection
                    #   glob above; copy_mosaic_helper_configs returns an empty list, not an error, for a
                    #   microscope with none available),
                    # naming rule: sequence_stem + hal_config_filename / shutter_filename / frame_table_filename
                    # (stem = "{kind}-{name}", kind in bits/cells/transit; hyphens delimit the prefix)
                    # + read_hal_flip_vertical, find_frame_table_for_hal_config, get_color_frame_indices,
                    #   get_all_color_frame_indices (every z-frame index for one color, not just the
                    #   single mid-z one get_color_frame_indices returns -- used by analysis/ffc.py's
                    #   "single_fov_all_frames" FFC sample-selection strategy, since vignetting doesn't
                    #   depend on z)
                    # + reconstruct_frame_table (inverse: hal+shutter XML -> frame table) and its parsers
                    #   read_shutter_reference, parse_z_offsets, parse_shutter_events
    positions.py    # create_grid_positions, generate_scanning_path, filter_scanning_path, close_scanning_path,
                    # load_hole_polygons (also reassembles hole{n}_island{m}.txt companions -- see mosaic.py --
                    #   into interior rings of that hole's Polygon, via Polygon(coords, holes=[...])), get_path_stats,
                    # find_exterior_fovs (FOVs on the exterior of an imaged FOV grid -- the true outer
                    #   perimeter AND any hole's inner boundary -- used by analysis/ffc.py's
                    #   "exterior_grid" FFC sample-selection strategy. Queries real stage coordinates
                    #   directly via a KD-tree rather than snapping onto one shared integer grid index
                    #   (the private _grid_indices below): each tissue boundary/piece gets its OWN FOV
                    #   grid centred on that piece's own bounding-box midpoint (create_grid_positions),
                    #   so different pieces' grids are not in general phase-aligned -- a shared-grid-index
                    #   snap would misjudge adjacency exactly at a tissue-piece boundary; real-coordinate
                    #   KD-tree queries are correct regardless of any other piece's phase, so multi-tissue/
                    #   hole layouts are handled with no per-tissue-piece logic needed)
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
                    #   .msc file's own rounded objective line; each tile's own (x_um, y_um) is additionally
                    #   corrected by its objective's real per-objective (x_offset, y_offset), parsed from the
                    #   same .msc manifest's `objective,<name>,<um_per_pix>,<x_offset>,<y_offset>` line via
                    #   _parse_objective_offsets -- confirmed directly, on real data, that a previously-
                    #   uncorrected mosaic/high-mag-alignment-tile discrepancy exactly matched this already-
                    #   recorded-but-previously-ignored value; this offset is NOT a fixed hardware constant
                    #   (objectives are physically removed/reinstalled between users on this shared microscope),
                    #   so it is always read fresh from each experiment's own .msc file, never hard-coded)
                    #   + filter_tiles_by_objective (kept for explicit
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
                    #   create_round_info share that one pad-width function now), create_dave_config
                    #   (positions_dir= enables per-segment loops; every loop gets its own identically-
                    #   named loop_variable even when several rounds share one positions file -- Dave's
                    #   real v2Generator indexes a loop by its own name, so a movie's <variable_entry>
                    #   cannot alias a differently-named shared loop_variable declared elsewhere;
                    #   <change_directory> is emitted immediately before each imaging loop, never before
                    #   a preceding fluidics block, purely for XML readability -- a caller passing
                    #   round_info restricted to only the bits/hyb rows (e.g. a standalone "hybs" recipe,
                    #   see notebook 04 below) sets leading_fluidics=True to emit that slice's own
                    #   leading fluidics block before its first round, reusing the exact same Kilroy-
                    #   protocol-resolution/first_hyb_no_cleave logic a full cells+hybs round_info would
                    #   have produced as a side effect of the preceding (now absent) cells round),
                    #   dave_config_filename / dave_cells_config_filename / dave_focustest_config_filename
                    #   (single source of truth for each recipe's dave-{mic}-{cells|N hybs|focustest}-
                    #   {name}.xml name, shared by the notebook that writes it and the one that later
                    #   annotates it -- avoids globbing settings/dave-*.xml and picking the wrong file
                    #   when two acquisitions with different hyb counts share one settings/ folder),
                    #   annotate_dave_with_round_info, series_to_movie_name, get_hal_frame_count,
                    #   create_focus_test_dave_config (print_estimate=True by default -- reuses
                    #   estimate_dave_experiment/format_experiment_estimate exactly like create_dave_config,
                    #   since the estimator already parses the written XML generically and needs no
                    #   changes to handle a fluidics-free single-loop recipe: check-only mode correctly
                    #   reports ~0 s/0 B, since no <length> element exists on that mode's <movie>)
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
                    # than a dense fixed-width array — used by misc/measure_tissue_thickness_test.ipynb, which
                    # needs every intensity value's exact count, not a lossy fixed-bin approximation, so it
                    # can derive a mean, a percentile, a re-binned view, and a true-pixel count for any
                    # threshold from ONE cached read instead of recomputing/re-reading pixels for each),
                    # save_channel_counters/load_channel_counters (persist/reload — ragged per-z arrays,
                    # stored as numpy object arrays, needing allow_pickle=True to read back),
                    # counter_mean/counter_percentile/rebin_counter/tpc_from_counter (derive a mean /
                    # percentile / re-binned histogram over arbitrary bin_edges / true-pixel count from a
                    # (values, counts) Counter — pure arithmetic, no raw pixel re-read), 
                    # tpc_profile_from_counters (per-z true-pixel-count profile purely from a
                    # compute_channel_counters() result — no disk read at all once Counters are cached),
                    # _summarize_tpc_profile (shared helper: given every z's true-pixel count, returns
                    # z_first_um/z_last_um — shallowest/deepest z with signal, either None if none passed —
                    # and is_contiguous, False if signal turned off and back on somewhere in between; real
                    # data showed tissue signal isn't always monotonic with depth, so every z must be
                    # counted rather than stopping at the first failure) — FOV-level analysis
    round.py        # create_mosaic, load_thumbnails_for_round — round-level mosaic from
                    #   pre-made per-FOV thumbnails (no flat-field correction, independent
                    #   per-tile contrast); create_mosaic_ffc, load_raw_frames_for_round —
                    #   flat-field-corrected sibling pipeline (see ffc.py) that reads RAW
                    #   per-FOV frames, divides out a per-pixel FFC field, crops the FOV's
                    #   overlap border, and applies ONE shared contrast stretch across the
                    #   whole assembled canvas instead of per-tile; kept as separate
                    #   functions (not a flag on create_mosaic) since existing callers of
                    #   create_mosaic rely on its {fov_id: uint8 array} contract unchanged.
                    #   Both share _layout_tiles (private) for identical tile placement.
    ffc.py          # Flat-field correction (FFC) for round mosaics (analysis/round.py's
                    #   create_mosaic_ffc consumes its output; does NOT touch the per-FOV
                    #   analyze_file/create_thumbnail pipeline). Computed once per
                    #   experiment per color (vignetting is a fixed optical property, not
                    #   per-round), cached via ProgressTracker's ffc_field_path/is_ffc_done/
                    #   mark_ffc_done. compute_and_cache_ffc(config, metadata, tracker,
                    #   color) is the entry point scheduler.build_round_mosaics calls
                    #   inline (idempotent -- cheap after the first call, since by the time
                    #   a round's mosaics are built every raw file FFC needs is already
                    #   guaranteed on disk, per ProgressTracker.all_fovs_done_for_round --
                    #   no separate SLURM job/array script needed). Three interchangeable
                    #   FOV/frame selection strategies feed one shared
                    #   compute_ffc_field_for_color(samples: List[(path, frame_idx)], ...)
                    #   (config.ffc_fov_selection_strategy): "exterior_grid" (default --
                    #   select_ffc_exterior_fovs, via acquisition.positions.
                    #   find_exterior_fovs), "emptiest_stats" (select_emptiest_fovs, ranks
                    #   FOVs by already-computed analysis/stats/*.csv mean+std -- no new
                    #   raw reads just to pick candidates), "single_fov_all_frames" (one
                    #   near-empty FOV's every z-frame of a color, via
                    #   select_all_frames_of_fov + acquisition.configs.
                    #   get_all_color_frame_indices -- vignetting doesn't depend on z).
                    #   Which strategy/minimum-sample-count is best is an open empirical
                    #   question investigated by notebooks/misc/
                    #   investigate_ffc_sample_size.ipynb, not decided in this module.
                    #   Also: apply_ffc, save_ffc_field/load_ffc_field (.npz cache),
                    #   compute_mosaic_crop_px(config) (generic overlap-border crop width
                    #   from image_size_px/non_overlap_fraction, not a hardcoded pixel
                    #   count), resolve_ffc_reference_round (per-color, since different
                    #   rounds can use different colors -- e.g. a cells round's 405 vs. a
                    #   bits round's 750/650/560).
    stage_z.py      # stage-z drift QC: off_path_for/read_off_file (HAL's per-movie ``.off``
                    #   focus-lock sidecar — same directory/stem convention as ``.inf``, whitespace-
                    #   delimited, one row per frame, column ``stage-z``); read_off_file_if_ready
                    #   (safe wrapper -- HAL creates the ``.off`` file before writing any/a complete
                    #   last row, so a reader running DURING acquisition can see it empty or
                    #   truncated; returns None instead of raising in that case, identical to "not
                    #   written yet" -- stage_z_summary_for_fov/focus_lock_summary_for_fov and
                    #   04_create_dave_config.ipynb's focus-lock-test read-back cell all go through
                    #   this rather than calling read_off_file directly, so a still-being-written
                    #   file never crashes a many-FOV loop), summarize_stage_z (first/min/max +
                    #   all_same, since the focus lock should hold stage-z constant for a whole
                    #   stack), update_stage_z_cache (extends an on-disk CSV cache with only
                    #   not-yet-read (round, FOV, series) combinations, so a many-thousand-FOV
                    #   experiment's ``.off`` files are each read at most once; coerces the cached
                    #   ``all_same`` column back to real bool dtype via _coerce_bool_column on both
                    #   load and save -- a bool column doesn't reliably survive a to_csv/read_csv
                    #   round-trip, and ``~`` on a non-bool column does bitwise, not logical,
                    #   negation, silently corrupting a ``cache[~cache["all_same"]]``-style filter
                    #   into a KeyError rather than a wrong answer), round_label/assign_x_positions
                    #   (``x = fov_id`` -- every round overlays at the SAME x position rather than
                    #   laid out back-to-back, so round-to-round drift at a given FOV shows as a
                    #   vertical offset between per-round lines, not a left-to-right shift) — see
                    #   notebooks/during_imaging/stage_z_drift.ipynb
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
                    #   fov_submitted/round_mosaic_submitted (cluster SLURM bookkeeping, hold the job id)
  progress_display.py # ProgressReporter — reusable, dependency-free live console/notebook progress +
                    #   ETA display for any long-running per-item loop (n/total, percent, a text bar,
                    #   elapsed, ETA extrapolated from the average per-item rate seen so far); wrap an
                    #   iterable directly (`for x in ProgressReporter(len(items), "label").wrap(items)`)
                    #   or drive update()/done() manually. Distinct from progress.py's ProgressTracker,
                    #   which persists completion via on-disk sentinels across separate runs — this is
                    #   purely a live display for a loop running right now, no persistence — meant for
                    #   notebooks that don't have their own, e.g. misc/measure_tissue_thickness_test.ipynb's
                    #   histogram backfill loop.
  scheduler.py      # FOVScheduler (continuous, parallel process-pool), RoundScheduler,
                    #   ExperimentScheduler — main analysis loops; also exports the shared
                    #   build_fov_task_kwargs/build_round_mosaics/resolve_round_flip_y/
                    #   resolve_round_color_frame_indices/source_dirs_for_round functions the
                    #   schedulers and the cli_*.py cluster scripts both call
  transfer.py       # transfer_round (per-round → NAS, one background thread per round; destination is
                    #   dest_root/relative_to_data_root(src) — e.g. dest_root/data/hybs/H01 — NOT dest_root/
                    #   src.name, so TRANSFER_DEST ends up a full mirror of SAMPLE_DIR rather than a flat
                    #   dump of per-round folder names), relative_to_data_root (strips everything before a
                    #   path's own 'data' component — the shared logic behind that destination choice),
                    #   mirror_tree (incremental data_dir → 2nd drive, background thread) / mirror_dir_sync
                    #   (same copy logic, but synchronous — for a notebook cell that wants to see a one-off
                    #   folder sync finish, e.g. data/mosaic10x or the static MERci/merlin/fishtank folders,
                    #   before moving on) — robocopy/shutil either way.
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
      merfish_multi_z/ # variable-z-per-FOV variant of merfish/ (own 10-notebook sequence, see
                     #   its own README.md and below — NOT part of the shared 8-notebook
                     #   tumor/lineage_tracing template) — for tissue whose depth varies enough
                     #   across the coverslip that imaging every FOV to one fixed z-depth wastes
                     #   time/disk on thin regions. Images a full-depth DAPI (cells) round first,
                     #   measures each FOV's real tissue depth from it, buckets FOVs into a few
                     #   z-depth tiers (quantile binning), then images each bits round to only the
                     #   depth its tier needs:
                     #   01_create_hal_config_and_shutters.ipynb      # TRIMMED: cells + transit only (bits
                     #                                                 #   deferred to 05 -- no single depth yet)
                     #   02_create_boundary_from_mosaic.ipynb         # unchanged (optional)
                     #   03_create_positions_from_boundaries.ipynb    # renumbered from merfish/'s 02
                     #   04_measure_tissue_thickness.ipynb            # NEW -- a lean, production subset of
                     #                                                 #   notebooks/misc/measure_tissue_
                     #                                                 #   thickness_test.ipynb (the full
                     #                                                 #   exploratory/R&D notebook: only the
                     #                                                 #   TPC z_last calculation, a fixed
                     #                                                 #   1um margin, a z_last-only heatmap,
                     #                                                 #   theoretical-only time/data savings,
                     #                                                 #   and one verification mosaic (below)
                     #                                                 #   -- no texture-profile/GIF diagnostics;
                     #                                                 #   run after the cells round; exports a
                     #                                                 #   per-FOV z table (metadata/
                     #                                                 #   z_per_fov_table.csv). Section 4 (the
                     #                                                 #   heaviest read step) has a
                     #                                                 #   USE_SLURM_ARRAY option, same
                     #                                                 #   convention as the sections below.
                     #                                                 #   Z_MAX_TRIMMED_UM (the cap on
                     #                                                 #   z_needed_um = z_last_um +
                     #                                                 #   Z_MARGIN_UM) defaults to this round's
                     #                                                 #   own actual max imaged z for
                     #                                                 #   CHANNEL_NM -- the true physical
                     #                                                 #   ceiling -- not z_last_um.max() +
                     #                                                 #   Z_MARGIN_UM, which could otherwise
                     #                                                 #   push a FOV's z_needed_um past a z
                     #                                                 #   that was never actually imaged. A
                     #                                                 #   verification mosaic renders each
                     #                                                 #   FOV's actual frame at the largest
                     #                                                 #   available z-step at or below its own
                     #                                                 #   z_needed_um (floor, not nearest, so
                     #                                                 #   the rendered/trimmed depth never
                     #                                                 #   overshoots); FOVs with no detected
                     #                                                 #   signal at all are still rendered (at
                     #                                                 #   the round's deepest available z) and
                     #                                                 #   flagged with a white border, so this
                     #                                                 #   mosaic can be used to confirm they are
                     #                                                 #   real tissue-free (e.g. border) FOVs
                     #                                                 #   before notebook 05 assigns them a tier.
                     #   05_create_hal_config_and_shutters_multi_z.ipynb # NEW: buckets FOVs into N_TIERS z-depth
                     #                                                 #   tiers from notebook 04's z table; writes one
                     #                                                 #   bits hal_config+shutter per tier into
                     #                                                 #   SAMPLE_DIR/multi_z/ (combined folder --
                     #                                                 #   HAL resolves <shutters> relative to its own
                     #                                                 #   xml_directory, so tier hal_config+shutter
                     #                                                 #   stay together rather than split into
                     #                                                 #   separate hal_configs/ and shutters/ folders);
                     #                                                 #   tags each positions-file FOV with its tier's
                     #                                                 #   hal_config stem via a 3rd column
                     #   06_create_round_info.ipynb                   # renumbered from merfish/'s 03; uses the
                     #                                                 #   DEEPEST tier as the representative bits
                     #                                                 #   hal_config; tags bits rows
                     #                                                 #   tissue_thickness="multi" + z_lengths (every
                     #                                                 #   tier's frame count, JSON-encoded ascending)
                     #   07_create_dave_config.ipynb                  # renumbered from merfish/'s 04; no functional
                     #                                                 #   changes -- create_dave_config already skips
                     #                                                 #   the static per-movie <length>/<parameters>
                     #                                                 #   for a tissue_thickness="multi" round (the
                     #                                                 #   positions file's own 3rd column supplies the
                     #                                                 #   real per-FOV values instead -- requires the
                     #                                                 #   patched storm_control Dave in
                     #                                                 #   ../../misc/dave_multi_z/, outside this repo)
                     #   08_create_data_organization.ipynb            # renumbered from merfish/'s 05; picks the bits
                     #                                                 #   frame table with the MOST frames among all
                     #                                                 #   frame-table-bits-*.csv matches (the deepest
                     #                                                 #   tier), so MERlin's declared z-range covers
                     #                                                 #   every FOV
                     #   09_create_experiment_info.ipynb              # renumbered from merfish/'s 06; bits hal_config
                     #                                                 #   lookup (exposure time) checks both settings/
                     #                                                 #   and multi_z/
                     #   10_create_merlin_scripts.ipynb                # renumbered from merfish/'s 07; passes
                     #                                                 #   --allow-ragged-z-stacks to the slurm submit
                     #                                                 #   script whenever round_info.csv has any
                     #                                                 #   tissue_thickness="multi" row
                     #   (four levels deep -> MERCI_DIR = ...parent.parent.parent.parent, same as merfish/)
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
    07_cluster_submit_analysis.ipynb               # run ON a cluster login/transfer node (after you've moved data
                                                   #   from the microscope/NAS to cluster storage yourself, e.g.
                                                   #   Globus/FileZilla): discovers pending FOVs/rounds and submits
                                                   #   SLURM array jobs (cli_analyze_fov.py / cli_build_round_mosaic.py,
                                                   #   via acquisition/cluster_submit.py) to do the QC analysis
                                                   #   01/02 would otherwise do locally -- 01/02 remain supported
                                                   #   for local (same_drive/mirror_drive) projects; 07 is the
                                                   #   cluster-QC alternative for projects that would rather move
                                                   #   analysis off the microscope computer entirely
  during_imaging/   # Live QC notebooks meant to be watched in real time WHILE HAL/Dave
                    #   is actively acquiring (as opposed to analysis/'s schedulers, which
                    #   run continuously in the background, or its one-shot review notebooks)
    stage_z_drift.ipynb                            # one-shot QC (moved from analysis/08_stage_z_drift.ipynb --
                                                   #   same notebook, unchanged logic): reads each FOV's ``.off``
                                                   #   focus-lock sidecar's stage-z column (analysis/stage_z.py,
                                                   #   tolerating a sidecar HAL has created but not finished
                                                   #   writing yet -- safe to run DURING acquisition),
                                                   #   caching results to analysis/cache/stage_z_drift/ so
                                                   #   re-runs only read newly-written ``.off`` files, then
                                                   #   plots the first-frame stage-z value as one line per
                                                   #   round, all overlaid at the SAME x position (``fov_id``)
                                                   #   rather than laid out back-to-back, so drift at a given
                                                   #   FOV across rounds reads as a vertical spread between
                                                   #   differently-colored lines
    imaged_fovs.ipynb                              # live acquisition-progress map: reuses prepare_imaging/
                                                   #   <variant>/02_create_positions_from_boundaries.ipynb's own
                                                   #   background (the real Steve low-mag mosaic photo when
                                                   #   boundaries came from 02_create_boundary_from_mosaic.ipynb,
                                                   #   else the same schematic tissue-boundary outline) and draws
                                                   #   one square per planned FOV, filling a square in as that
                                                   #   FOV's image file actually appears on disk -- polls faster
                                                   #   than one FOV takes to acquire (ExperimentMetadata.
                                                   #   series_for_round(...).resolve_path(...).exists(), no image
                                                   #   reads) so no FOV is missed; ROUND_ID=None auto-detects
                                                   #   whichever round currently has SOME but not ALL FOVs written.
                                                   #   If nothing is actively in progress (e.g. run during the
                                                   #   fluidics gap between one round finishing and the next
                                                   #   starting -- fluidics is strictly between rounds' imaging
                                                   #   loops, never mid-round, so the next round has zero files at
                                                   #   that point and is otherwise invisible to this scan), it
                                                   #   points at the round AFTER the most recently completed one
                                                   #   instead (unless that's already the last round), so the
                                                   #   notebook is ready and waiting rather than showing an
                                                   #   already-finished round as "done"; falls back to the first
                                                   #   round if nothing anywhere has been imaged yet. A FOV is
                                                   #   additionally flagged (different fill
                                                   #   color) if its ``.off`` sidecar's ``good-offset`` column is
                                                   #   1 in FEWER than ``MIN_GOOD_OFFSET_FRAMES`` frames (default
                                                   #   1 -- flags only a FOV where the two-spot focus lock was
                                                   #   never found at all, i.e. good-offset is 0 for EVERY frame;
                                                   #   raise it to also flag a weak/marginal lock found only
                                                   #   briefly). The default is deliberately NOT "any 1->0
                                                   #   transition" -- that's the normal case of losing lock once a
                                                   #   z-sweep goes past the lock's own tracking range, which
                                                   #   flips most FOVs' own good-offset from 1->0 partway through
                                                   #   and should not be flagged. This heuristic was chosen after
                                                   #   investigating whether storm_control's Dave/HAL could log a
                                                   #   real focus-lock "warning" to a file instead: they don't
                                                   #   reliably -- the only place that information appears on disk
                                                   #   is an incidental byproduct of Dave's generic, rotating debug
                                                   #   log (``<data_dir>/logs/dave_N.out``, capped/rotated,
                                                   #   filename not fixed, no per-FOV tag on the record) -- too
                                                   #   fragile to poll from an unattended notebook (see
                                                   #   prompt_history/ for the full investigation). Live redraw
                                                   #   uses plain IPython.display.clear_output+display(fig) on a
                                                   #   loop (no ipywidgets dependency) -- interrupt the kernel to
                                                   #   stop; the last drawn state is kept and saved to
                                                   #   ``SAMPLE_DIR/figures/`` -- deliberately NOT
                                                   #   ``analysis/figures/`` like every other notebook
                                                   #   (NOTEBOOK_GUIDELINES.md #6), since this is a live view meant
                                                   #   to be checked at a glance during acquisition, not filed
                                                   #   away with post-hoc QC figures.
    round_mosaics.ipynb                            # live quick-look mosaic, one state table + one loop
                                                   #   driving all three use cases (on-demand/catch-up/live --
                                                   #   previously three separate notebook sections, unified
                                                   #   after the split version proved awkward to reorder and hid
                                                   #   a real cross-section NameError). One frame per FOV per
                                                   #   real color (EXCLUDED_COLORS, default just 488nm -- HAL's
                                                   #   own bead/focus-lock reference channel, always at a fixed
                                                   #   bead z regardless of TARGET_Z_UM, not real tissue signal;
                                                   #   set to [] to include it again), near a fixed TARGET_Z_UM
                                                   #   (default 10.0 -- a real stage z in um, same convention as
                                                   #   every other z parameter in this package; resolved per
                                                   #   round/color via each round's own frame table, warning if
                                                   #   the nearest available z is >5um away from the request) --
                                                   #   no z-stack read, deliberately light enough to run
                                                   #   continuously alongside a real acquisition reading
                                                   #   straight off the NAS.
                                                   #
                                                   #   SELECTED_ROUNDS ("all" default, or a list mixing round
                                                   #   labels like "cells" -- resolved by imaging_type, same
                                                   #   convention as correct_camera_rotation.ipynb's own
                                                   #   ROUND_IMAGING_TYPE -- and/or explicit imaging_round
                                                   #   numbers) + LIVE_LOOP (True default) together subsume the
                                                   #   old three modes: a single round + LIVE_LOOP=False = old
                                                   #   "on-demand specific round"; "all" + LIVE_LOOP=False = old
                                                   #   "catch-up pass" (one pass over whatever's already fully
                                                   #   imaged); "all" + LIVE_LOOP=True (default) = old "live
                                                   #   loop," watching/advancing across every round for the
                                                   #   whole experiment. Section 6 builds a small state table
                                                   #   (one row per selected round: imaged_fovs, processed_fovs
                                                   #   -- FOVs with a cached thumbnail for EVERY one of that
                                                   #   round's colors, not just one -- and total_fovs); section
                                                   #   7's loop rebuilds that table every cycle and processes any
                                                   #   round with processed_fovs < imaged_fovs, in the order
                                                   #   SELECTED_ROUND_IDS lists them, stopping once every
                                                   #   selected round is both fully imaged AND fully processed
                                                   #   (or MAX_RUNTIME_MIN is exceeded) if LIVE_LOOP, else after
                                                   #   one pass.
                                                   #
                                                   #   SLURM/cluster parallelization (one array job per FOV,
                                                   #   mirroring 07_cluster_submit_analysis.ipynb) was considered
                                                   #   and deliberately deferred: that pattern exists in this
                                                   #   repo for a much heavier per-FOV task (analyze_file's full
                                                   #   multi-frame z-stack read, budgeted at 2h per SLURM task);
                                                   #   this notebook's per-FOV cost (one named frame + a cheap
                                                   #   thumbnail) is a fraction of that, and SLURM's own
                                                   #   submission/queue overhead would plausibly dominate a task
                                                   #   this light rather than speed it up -- sequential only for
                                                   #   now, revisit (or a local ProcessPoolExecutor, matching
                                                   #   FOVScheduler's own pattern, avoiding SLURM's overhead
                                                   #   entirely) only if a real bulk backlog proves too slow.
                                                   #
                                                   #   Section 5 ("shared helpers") holds every read/thumbnail/
                                                   #   mosaic function used by section 6/7, run before either --
                                                   #   also where flat-field correction (ENABLE_FFC, off by
                                                   #   default) is computed: FFC_N_FOVS (default 10) real
                                                   #   exterior FOVs (acquisition.positions.find_exterior_fovs)
                                                   #   per color, a small ONE-TIME cost (vignetting is fixed per
                                                   #   experiment, cached to analysis/cache/round_mosaics/
                                                   #   ffc_field_{color}nm.npz) reused for every round after. A
                                                   #   round already 100% imaged when it's first processed gets
                                                   #   FFC the full production way (analysis.round.
                                                   #   create_mosaic_ffc, real raw frames + one shared
                                                   #   whole-canvas contrast stretch); one still being imaged
                                                   #   gets it divided out frame-by-frame right after reading,
                                                   #   before independently thumbnailing (keeping a whole
                                                   #   in-progress round's raw frames in memory isn't
                                                   #   affordable) -- a stated quality tradeoff, and EITHER WAY
                                                   #   still writes the plain (non-FFC) thumbnail alongside the
                                                   #   raw-frame path specifically so `processed_fovs` has
                                                   #   something to detect -- confirmed directly that without
                                                   #   this, an FFC-enabled round could never register as
                                                   #   processed at all (build_round_mosaic's FFC branch never
                                                   #   touched THUMBNAILS_DIR) and LIVE_LOOP=True would loop on
                                                   #   it forever. Reads/thumbnails go to the shared
                                                   #   analysis/thumbnails/ location 01_fov_scheduler.ipynb also
                                                   #   uses (interoperable either direction, and always the
                                                   #   plain raw-frame convention regardless of ENABLE_FFC, so
                                                   #   the cache stays reusable if FFC is later toggled off);
                                                   #   mosaics save to SAMPLE_DIR/figures/ (same deliberate
                                                   #   exception as imaged_fovs.ipynb) under
                                                   #   round_mosaics.round{id:03d}_{color}nm.png -- a different
                                                   #   filename pattern than tracker.mosaic_path's
                                                   #   analysis/mosaics/round_{id:03d}_{color}nm_mosaic.png, so
                                                   #   this quick-look tool and analysis/02_round_scheduler.
                                                   #   ipynb's production (mid-z, optional FFC, round-complete-
                                                   #   only) mosaics can run at the same time without clobbering
                                                   #   each other. `CATCHUP_READ_DELAY_SEC` (default 0.02s, ~50
                                                   #   files/sec cap) paces FRESH reads only (a cached thumbnail
                                                   #   costs no sleep) -- there's no universal "safe NAS read
                                                   #   rate" (depends on that NAS's real hardware/network
                                                   #   capacity, which nothing here can measure), but HAL's own
                                                   #   write demand is small/steady and computable
                                                   #   (~image_size_px**2*2 bytes per exposure_time), well under
                                                   #   typical network bandwidth on bytes alone -- the real risk
                                                   #   this default guards against is IOPS/seek contention from
                                                   #   many small scattered reads landing on the same storage
                                                   #   HAL is sequentially writing to, not raw throughput; see
                                                   #   the notebook's own section 5 markdown for the full
                                                   #   reasoning and how to check/tune it empirically (real
                                                   #   file-write-mtime gaps, the same technique
                                                   #   misc/measure_tissue_thickness_test.ipynb section 8 uses).
                                                   #
                                                   #   Bulk preload + camera orientation + shared contrast (all
                                                   #   in section 5): a dedicated `tile_cache_path` PNG cache
                                                   #   (per round/color/FOV -- oriented + FFC-divided-if-enabled
                                                   #   + shared-contrast-stretched, i.e. exactly the placed
                                                   #   canvas pixels, distinct from THUMBNAILS_DIR's plain
                                                   #   processed-marker convention) lets an already-mostly-done
                                                   #   round (e.g. 666/1166 from a prior kernel session) reload
                                                   #   near-instantly with NO raw re-reads; `build_round_mosaic`
                                                   #   bulk-loads every cached tile with no per-tile redraw, shows
                                                   #   that accumulated state once, THEN reads/places any
                                                   #   remaining FOV one at a time with a redraw after each. Every
                                                   #   raw frame is re-oriented via `apply_microscope_orientation`
                                                   #   (MERlin's own transpose/flip_horizontal/flip_vertical
                                                   #   convention for `MICROSCOPE`, a section-2 parameter -- same
                                                   #   pattern as `misc/correct_camera_rotation.ipynb`) right
                                                   #   after reading; a cached FFC field is itself estimated from
                                                   #   UN-oriented raw pixels and oriented once (cheaper) --
                                                   #   averaging/smoothing/normalization/clipping all commute
                                                   #   exactly with a fixed transpose/flip, so this is
                                                   #   mathematically identical to orienting every sample frame
                                                   #   first. A single shared `(vmin, vmax)` contrast range per
                                                   #   (round, color) is estimated once from
                                                   #   `CONTRAST_SAMPLE_N_FOVS` (default 10) random INTERIOR
                                                   #   (non-exterior) FOVs' pooled histogram at `CONTRAST_LOW_
                                                   #   PCT`/`CONTRAST_HIGH_PCT` (default 5th/99th) percentile,
                                                   #   cached to `analysis/cache/round_mosaics/contrast_range_
                                                   #   round{r}_{color}nm.npz` -- every tile then uses this FIXED
                                                   #   range instead of an independent per-frame percentile, so
                                                   #   tiles stay visually consistent without needing the whole
                                                   #   round done first.
                                                   #
                                                   #   Redraw cost must stay independent of real mosaic size.
                                                   #   The tile-by-tile live-redraw design above was only ever
                                                   #   verified against 20 small fake FOVs; confirmed directly
                                                   #   that at LT060_sample_04's real production scale (1166
                                                   #   FOVs, a ~10600x6700 px canvas) matplotlib's own
                                                   #   `imshow`+draw of the FULL-resolution array cost ~6s and a
                                                   #   full-resolution PNG disk write ~3s -- EVERY time
                                                   #   `maybe_redraw` fired (up to every `LIVE_REDRAW_MIN_
                                                   #   INTERVAL_SEC`, default 0.5s), completely dominating
                                                   #   wall-clock time regardless of how fast the underlying FOV
                                                   #   reads were, so per-tile "live" updates in practice arrived
                                                   #   only every several seconds -- functionally the same
                                                   #   "crawls FOV by FOV" complaint the tile-cache/live-redraw
                                                   #   design was meant to fix, just not caught by a small-scale
                                                   #   test. Fixed by decoupling the two costs: the ON-SCREEN
                                                   #   preview is downsampled to `LIVE_PREVIEW_MAX_PX` first
                                                   #   (`_downsample_for_preview`, confirmed directly to cost
                                                   #   well under 0.1s regardless of real canvas size), so it
                                                   #   redraws genuinely live every `LIVE_REDRAW_MIN_INTERVAL_
                                                   #   SEC`; the FULL-RESOLUTION PNG actually saved to
                                                   #   `figures/` is only rewritten at the much coarser
                                                   #   `DISK_SAVE_MIN_INTERVAL_SEC` (default 20s) or on a forced
                                                   #   call (`show_round_mosaic`'s new `save_full_res` param),
                                                   #   bounding on-disk staleness without paying the expensive
                                                   #   write on every tile.
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
    stage_drift_beads.ipynb / stage_drift_dapi.ipynb  # measure same-microscope stage drift
                                                   #   since an already-imaged round (cells, hybs/H0X, ...) and
                                                   #   correct the whole positions file for it. Two sibling
                                                   #   notebooks, split because the registration channel choice is
                                                   #   a real fork, not just a parameter default: `_beads` registers
                                                   #   on the reference round's own auto-detected bead colour (e.g.
                                                   #   488) at its own bead_z (`acquisition.alignment.
                                                   #   select_bead_frame`); `_dapi` registers on an explicit
                                                   #   REGISTRATION_COLOR_NM (default 405/DAPI) at a user-specified
                                                   #   REGISTRATION_Z_UM instead, for when the bead channel proves
                                                   #   too dim/unreliable even after hot-pixel removal. Both share:
                                                   #   Part A -- pick the reference round (lists every round in
                                                   #   round_info.csv; REFERENCE_FRAME_TABLE_PATH can point at a
                                                   #   specific frame-table CSV instead of auto-resolving one from
                                                   #   the round's hal_config, for when metadata/ has drifted out
                                                   #   of sync with the real frame table used at acquisition time),
                                                   #   take FOVs FIRST_FOV..FIRST_FOV+N_FOVS-1 (in the original
                                                   #   experiment's global fov_id numbering; default 0..2) and
                                                   #   build a tiny single-colour "drift" round for just those
                                                   #   positions -- a positions_*_drift.txt, a HAL config/shutter,
                                                   #   and a minimal single-loop (no-fluidics) Dave recipe -- all
                                                   #   written to SAMPLE_DIR/stage_drift/<reference round's own
                                                   #   data_dir subpath>/ (e.g. stage_drift/cells/,
                                                   #   stage_drift/hybs/H01/), a folder at the same level as
                                                   #   MERci/. The new round's series pattern/file numbering is
                                                   #   built fresh from dave.fov_pad_width(N_FOVS) and LOCAL
                                                   #   0-based indices (plus explicit fov_start/fov_pad row columns
                                                   #   for the patched Dave, misc/dave_multi_z/) -- NOT the
                                                   #   reference round's own (differently-sized) series pattern,
                                                   #   since Dave numbers each loop's movies with a counter that
                                                   #   resets to 0 and zero-pads only as wide as that loop's own
                                                   #   position count (confirmed against storm_control's
                                                   #   v2Generator.py, prompt_history/
                                                   #   2026_07_08_1557_investigate_dave_fov_index_range.md) --
                                                   #   reusing the original pad width silently predicted filenames
                                                   #   Dave never actually wrote. Run the Dave recipe on the
                                                   #   microscope, then Part B: registers each new frame onto its
                                                   #   reference frame via skimage.registration.
                                                   #   phase_cross_correlation (acquisition.alignment.phase_drift
                                                   #   -- the same primitive fishtank's align_experiments uses;
                                                   #   MERlin has no equivalent, reused rather than reimplemented)
                                                   #   after acquisition.alignment.remove_hot_pixels (ported from a
                                                   #   validated fix to the identical bug in this project's own
                                                   #   fishtank fork -- a fixed hot/dead camera pixel, identical in
                                                   #   every frame, dominates phase_cross_correlation when the real
                                                   #   signal is dim and pins the recovered shift to exactly
                                                   #   [0, 0]; local-median outlier removal catches isolated
                                                   #   pixel-scale spikes but not a broader fixed pattern like
                                                   #   vignetting), combines the per-FOV shifts into one
                                                   #   translation (median, reporting spread as a QC check --
                                                   #   DRIFT_SIGN flags that the pixel->stage-um sign convention is
                                                   #   unverified per microscope, same caveat as
                                                   #   alignment.compute_fov_drifts's own sign_x/sign_y), and
                                                   #   applies it to EVERY FOV in the experiment's positions file
                                                   #   (not just the N sampled), writing a drift-corrected
                                                   #   positions file for the next real imaging round. Each has its
                                                   #   own section-8 diagnostic: `_beads` compares bead frames
                                                   #   across cells + bits rounds (skipping any round with no
                                                   #   auto-detectable bead colour, rather than crashing); `_dapi`
                                                   #   instead shows the exact reference/new frame pair side by
                                                   #   side (a cross-round comparison doesn't apply -- bits rounds
                                                   #   typically carry no DAPI/tissue channel at all). Adds a
                                                   #   "drift" hal_config/shutter/frame-table kind (configs.py
                                                   #   _VALID_KINDS) alongside bits/cells/transit.
    verify_kilroy_protocol_consistency.ipynb       # verify a Kilroy config's protocols only reference defined
                                                   #   valve/pump commands; fuzzy-suggest + (after confirm) rewrite
                                                   #   mismatches in place (backup to *.bak), via kilroy.py helpers
    audit_disk_usage.ipynb                         # scan a list of shared-drive roots laid out as
                                                   #   {root}/{lab_member}/{sample_dir}, measure each sample
                                                   #   folder's size + creation date (disk_audit.py), and display
                                                   #   it sorted oldest-first and largest-first — to find whose
                                                   #   data to ask to be cleared off a shared microscope computer
    create_mosaic_gif.ipynb                        # animated GIF of one full mosaic (every FOV) per
                                                   #   z-step, for a chosen round + channel -- watch
                                                   #   image content change across the whole imaged
                                                   #   depth. GIF_Z_STRIDE subsamples the z-grid (reading
                                                   #   every FOV at every z-step is heavy, multi-hour-
                                                   #   scale I/O); USE_SLURM_ARRAY submits the read as a
                                                   #   SLURM array job instead of running sequentially
                                                   #   (cli_compute_gif_frame_thumbnails.py +
                                                   #   cluster_submit.build_gif_frames_array_script) --
                                                   #   generalizes measure_tissue_thickness_test.ipynb's
                                                   #   own (cells-round/DAPI-only) GIF section
    visualize_channel_zsweep.ipynb                 # standalone (no SAMPLE_DIR/round_info context needed,
                                                   #   unlike create_mosaic_gif.ipynb above) diagnostic: given
                                                   #   ONE image file (dax/tiff/zarr) + its frame table CSV,
                                                   #   builds (1) a side-by-side animated GIF, one panel per
                                                   #   channel, sweeping z from shallowest to deepest -- each
                                                   #   channel's own frame chosen by floor (largest z <= the
                                                   #   step, same convention as measure_tissue_thickness's
                                                   #   floor_z_position), so a channel imaged at only a single z
                                                   #   (e.g. a bead/DAPI reference frame) just repeats that one
                                                   #   frame every step; contrast stretched PER CHANNEL (not one
                                                   #   shared scale), since different lasers/dyes have very
                                                   #   different intensity ranges; and (2) a mean-intensity-vs-z
                                                   #   line plot, one line per channel. Frame reads cached under
                                                   #   OUTPUT_DIR/cache/ (defaults next to the image file) --
                                                   #   no USE_SLURM_ARRAY option, since a single file's frame
                                                   #   count is small enough that one notebook kernel handles it
    investigate_ffc_sample_size.ipynb              # how many FOVs/frames does flat-field correction
                                                   #   (analysis/ffc.py) actually need to converge? Builds a
                                                   #   reference FFC field from every grid-exterior FOV of a
                                                   #   chosen round/color, then compares it (median/p95 percent
                                                   #   difference) against fields built from increasing N of
                                                   #   the emptiest-by-stats FOVs (one frame each), and
                                                   #   separately against fields built from an increasing
                                                   #   number of z-frames of a single near-empty FOV --
                                                   #   answering whether many partially-filled exterior FOVs or
                                                   #   one near-empty FOV's full z-stack converges faster, and
                                                   #   with how few samples. Plots both convergence curves and
                                                   #   reports the smallest sample count under a tolerance;
                                                   #   findings feed a manual choice of
                                                   #   ExperimentConfig.ffc_fov_selection_strategy/
                                                   #   ffc_min_samples' production defaults -- not automated by
                                                   #   the notebook itself.
    measure_tissue_thickness_test.ipynb            # per-FOV map of where (z, µm) real tissue signal
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
                                                   #   pixel-count (TPC) profile directly from its cached
                                                   #   Counter (analysis.fov.tpc_profile_from_counters /
                                                   #   tpc_from_counter), reporting z_first_um/z_last_um (the
                                                   #   shallowest/deepest z with TPC > TPC_THRESHOLD) and
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
  tests/           # Diagnostic/recovery notebooks built to investigate or fix ONE specific
                    #   real issue on a real experiment, kept afterward as a worked-example
                    #   template for the same class of problem elsewhere -- not part of the
                    #   numbered prepare_imaging pipeline or a standing analysis tool. Two
                    #   levels deep like analysis/misc/during_imaging (MERCI_DIR = ...parent.parent).
    fix_mosaic_shift_missing_fovs.ipynb            # recovers from a skipped/mistimed 10x-vs-real-
                    #   imaging-objective calibration shift: the usual workflow scans a low-mag
                    #   (e.g. "10x") Steve mosaic, images a handful of FOVs at the real imaging
                    #   objective (e.g. "60x") over part of it to measure a fixed stage-calibration
                    #   offset, shifts the low-mag mosaic by that offset, THEN derives the tissue
                    #   boundary/FOV grid -- if the shift is skipped or applied too late (the
                    #   boundary already saved from the un-shifted mosaic), every FOV grid built
                    #   from it ends up offset from where the tissue actually is.
                    #   **Root cause since fixed upstream**: this shift turned out to already be
                    #   recorded, per-objective, in the mosaic's own `.msc` manifest, and
                    #   `acquisition.mosaic.load_steve_mosaic` now parses and applies it
                    #   automatically (see that module's entry above) -- this bug class cannot recur
                    #   for any FUTURE experiment. This notebook remains the one-off remediation for
                    #   LT060_sample_04, whose boundary was already generated before the fix existed;
                    #   sections 1-2 below manually RECONSTRUCT the pre-fix ("uncorrected") tile
                    #   positions (subtracting the same `.msc`-recorded offset, read once via
                    #   `acquisition.mosaic._parse_objective_offsets` -- single source of truth, never
                    #   hand-duplicated) purely to keep the original before/after illustration; every
                    #   tile loaded from `load_steve_mosaic` itself is already correctly positioned.
                    #   (1)/(2) composite BOTH objectives together (`acquisition.mosaic.
                    #   load_steve_mosaic`'s own mixed-objective support, cached locally --
                    #   `analysis/cache/fix_mosaic_shift_missing_fovs/steve_tiles.pkl`, since
                    #   reading many small per-tile files over a network acquisition drive is slow
                    #   regardless of total data volume) at the whole-mosaic scale, reconstructed-
                    #   unshifted then as actually loaded (corrected); `normalize_tile_intensities`
                    #   independently rescales each objective's own tiles onto a shared display range
                    #   first (`LOW_MAG_DISPLAY_PCT`/`HIGH_MAG_DISPLAY_PCT`), since the two objectives'
                    #   very different exposures make a shared-percentile stretch over raw values
                    #   unreadable (the high-mag patch saturates). **Known limitation, confirmed by
                    #   directly comparing the two rendered images**: at `WORKING_PIXEL_UM` (matching
                    #   the segmentation canvas), a shift of a few hundred um is visually
                    #   imperceptible against the whole mosaic's own scale -- these two whole-mosaic
                    #   composites read as confirmatory context, not the decisive evidence. A THIRD,
                    #   zoomed composite immediately follows: cropped to the high-mag tiles' own
                    #   bounding box (+ margin) at a finer `ZOOM_PIXEL_UM`, same two-objective
                    #   composite/normalization, before vs. after side by side -- confirmed on real
                    #   LT060_sample_04 data to show the actual alignment difference clearly (the
                    #   high-mag patch's texture visibly integrates into the surrounding tissue only
                    #   in the "after" panel). Step 3's real FOV-perimeter overlay (below) is what most
                    #   clearly shows the CURRENT grid's own misalignment (as opposed to whether the
                    #   shift itself is right, which the zoomed composite above addresses). (3) overlays
                    #   the CURRENT (already-imaging) positions file's own FOV PERIMETERS (yellow
                    #   squares, not center points) on the corrected low-mag-only canvas (real,
                    #   un-normalized values -- the same canvas segmentation reads in step 4) --
                    #   verified on real LT060_sample_04 data to show a clear, visible offset between
                    #   the current grid and the corrected tissue. (4) re-runs `segment_mosaic_tissue`
                    #   on the corrected canvas with the exact parameters this sample's own local
                    #   `02_create_boundary_from_mosaic.ipynb` was run with (copied verbatim, not
                    #   re-derived), then finds FOV positions in the new tissue boundary using the
                    #   SAME dense grid (`create_grid_positions`/`generate_scanning_path` centered on
                    #   the OLD boundary's own bounding box, sized to cover both regions) the CURRENT
                    #   positions file's own grid came from, filtered twice via `filter_scanning_path`
                    #   (once against the OLD boundary as a sanity check -- reproduced the real
                    #   current positions file's count exactly on LT060_sample_04 -- once against the
                    #   NEW/corrected boundary) -- NOT an independently re-centered grid for the new
                    #   boundary, since the shift isn't an exact integer number of FOVs and a
                    #   fresh grid would share almost no exact coordinates with the current one even
                    #   where they truly overlap; (5) overlays the OLD vs. NEW tissue BOUNDARY outlines
                    #   (not FOV points -- a shape-level sanity check); (6) classifies every FOV as
                    #   already-covered, MISSING (in NEW but not OLD -- real tissue the old grid never
                    #   imaged), or UNNECESSARY (in OLD but not NEW -- imaging time the old grid spent
                    #   off the corrected tissue) by EXACT coordinate membership (both filtered sets come
                    #   from the same dense grid, so a shared FOV is bit-identical, not just closely
                    #   overlapping -- no distance-based fuzzy matching needed), drawn as real FOV
                    #   perimeter squares over both boundary outlines on a blank background; verified on
                    #   real data to isolate a coherent tissue-edge strip, not scattered noise. Section
                    #   10 immediately redraws the SAME section-9 annotations on top of the real
                    #   corrected mosaic image (`shifted_canvas`, converting every polygon/FOV coordinate
                    #   to that canvas's own pixel space) instead of a blank background, since
                    #   `old_boundary_polygon`/`new_tissue_polygon`/`new_grid_positions` all already
                    #   share ONE real-stage-coordinate frame (the correction is exactly what brings
                    #   mosaic-derived coordinates into it). Section 11: an old-to-new FOV RENUMBERING
                    #   table (`positions/fov_renumbering_{tag}.csv`) -- the alternative to appending
                    #   missing FOVs (sections 12-13) is to image every remaining round directly from
                    #   the NEW positions file instead, which changes which FOV INDEX a given physical
                    #   location has; maps every OLD FOV index to its NEW index (or flags it dropped,
                    #   for the UNNECESSARY ones) so already-imaged and newly-imaged rounds can still be
                    #   combined under one consistent numbering later (e.g. for MERlin decoding).
                    #   (7) re-orders just the missing subset into its own short-travel loop -- tried
                    #   and MEASURED (`get_path_stats`) two alternatives first: keeping the new grid's
                    #   own filtered order, and a per-column boustrophedon re-sort of the subset alone,
                    #   neither of which reliably beat the other (a single lattice column can itself
                    #   contain more than one disconnected run of missing FOVs) -- a greedy
                    #   nearest-neighbor walk (starting from the point closest to the current positions
                    #   file's own last FOV, so the transit into the new loop is also short) won
                    #   decisively on real data; (8) appends the re-ordered missing FOVs after the
                    #   current positions and writes a NEW `_added`-suffixed positions file -- never
                    #   overwrites the file actually being imaged from.
                    #   Sections 14-17, appended analyses (independent of the shift-correction pipeline
                    #   above, reusing its already-loaded `current_positions`/`shifted_canvas`/
                    #   `low_tiles_uncorrected`/`missing_coords`): (14) how many OLD FOVs are empty vs.
                    #   have real tissue signal -- reads each FOV's own cells-round frame ONCE (cached
                    #   locally, since ExperimentMetadata's eager multi-round path resolution proved
                    #   impractically slow here -- round_info.csv's `dir` column records a
                    #   now-unreachable absolute path from a different machine, see `common/
                    #   metadata.py`'s entry above -- so this section resolves the cells round's own
                    #   file pattern directly against `SAMPLE_DIR` instead of constructing a full
                    #   ExperimentMetadata), classifies via a threshold estimated from the per-FOV
                    #   signal histogram's own bimodal structure (`acquisition.mosaic.
                    #   _estimate_bimodal_threshold`, reused directly). Confirmed directly on real
                    #   LT060_sample_04 data that this peak/valley estimator can fail even on a genuinely
                    #   bimodal distribution when the two classes are very differently SIZED (a tall,
                    #   narrow empty-FOV spike next to a much larger, broad, unevenly-shaped tissue-signal
                    #   hump) -- falling back to the naive median in that case produced a meaningless,
                    #   suspiciously-exact 50/50 split (caught by eye, not assumed correct just because it
                    #   ran without error); the fallback is `skimage.filters.threshold_otsu` instead
                    #   (maximizes between-class variance directly, not shape-dependent), which produced a
                    #   visually well-justified threshold sitting right at the real valley. Either way,
                    #   shown as a histogram for visual confirmation before trusting it. Section 15
                    #   overlays every already-imaged (OLD) FOV's own perimeter (yellow dotted) plus the
                    #   subset classified EMPTY at the current `EMPTY_THRESHOLD` (red dotted) on the OLD
                    #   mosaic (`old_canvas`, built from `low_tiles_uncorrected` since `current_positions`
                    #   lives in that same pre-fix, real-stage-coordinate frame, NOT the corrected frame
                    #   `shifted_canvas` uses) -- self-contained (reuses `old_stats_df`'s already-cached
                    #   per-FOV values, no re-read), so overriding `EMPTY_THRESHOLD` and re-running just
                    #   this cell interactively finds which currently-imaged FOVs are unambiguously empty
                    #   and safe to discard. (16) for MISSING FOVs specifically (never actually imaged,
                    #   so no real per-FOV data exists) -- samples the mean LOW-MAG MOSAIC intensity at
                    #   each one's own footprint as a proxy, classified against the SAME `THRESHOLD`
                    #   already used for tissue segmentation, for consistency; (17) if hybs 1-5 (bits
                    #   1-10) were lost, which LT2 codebook genes are affected and how many are
                    #   error-correctable -- parses `data/configs/merlin/codebooks/LT2v0_codebook.csv`
                    #   using MERlin's own OLD-FORMAT codebook-loading logic (`merlin.data.codebook.
                    #   Codebook.__init__`, copied from the real source, not guessed, since a plain
                    #   `pandas.read_csv` can't parse this file's header), verifies directly (not
                    #   assumed) that every barcode has weight 4 and the codebook's minimum pairwise
                    #   Hamming distance is 4 (a genuine MHD4 code), then classifies each gene by how
                    #   many of its 4 "on" bits fall in the dead range: 0 = unaffected, exactly 1 =
                    #   error-corrected (MERFISH's standard single-bit-dropout recovery, since
                    #   distance-4 codewords guarantee a 1-bit-corrupted pattern is still closest to its
                    #   true code), 2+ = lost.
                    #   Sections 18-20: the alternative to `_added` (appending missing FOVs to the
                    #   CURRENT positions file, sections 11-13) -- a FRESH positions file built
                    #   entirely from the corrected boundary. (18) writes `new_grid_positions`
                    #   (section 7 -- already "the same shared grid, filtered against the shifted
                    #   boundary", not an independently re-centered one) directly to
                    #   `positions_{tag}_shift.txt`; row order inherits `dense_path`'s own
                    #   boustrophedon order, NOT notebook 02's final `close_scanning_path`-style
                    #   return-leg closing (same documented gap as `old_grid_positions`, section 10).
                    #   (19) overlays BOTH `_added` and `_shift` as real FOV perimeter squares on the
                    #   real corrected mosaic (`shifted_canvas`) -- confirmed on real data that `_added`
                    #   extends slightly further at the tissue edges (its 200 UNNECESSARY FOVs,
                    #   already-imaged but outside the corrected boundary) while `_shift` tracks the
                    #   true boundary tightly. **Found and fixed a real, non-deterministic matplotlib
                    #   quirk here**: the usual "invisible `ax.plot([], [], label=...)`" trick for
                    #   giving un-labelable `Rectangle` patches a legend entry intermittently failed
                    #   ("No artists with labels found") when combined with `imshow` immediately
                    #   before it -- confirmed directly via isolated repro that the SAME exact code
                    #   sometimes works and sometimes doesn't (not a fixed, deterministic bug, and not
                    #   reproduced by any single isolated factor -- color name and linestyle kwarg
                    #   alone each worked fine independently); fixed by passing explicit
                    #   `matplotlib.lines.Line2D` objects straight to `ax.legend(handles=[...])`
                    #   instead of relying on an implicit empty-artist scan, which sidesteps the
                    #   flakiness entirely. (Every OTHER section's own `ax.plot([], [], label=...)`
                    #   calls -- 6/9/10/15 -- use the same fragile pattern and are equally exposed to
                    #   this, just didn't happen to trip it during this session's runs; not
                    #   retrofitted since not requested, but worth knowing.) (20) a scatter of every
                    #   `_added` FOV's own row index against its matching `_shift` row index (EXACT
                    #   coordinate match, same convention as section 11's renumbering table) --
                    #   confirmed on real data this bijects cleanly: all 1162 `_shift` FOVs are
                    #   reached, the 200 unmatched `_added` FOVs are exactly the UNNECESSARY count from
                    #   section 9. The plot itself shows two visually distinct regimes: `_added`'s
                    #   first ~1166 rows (the original `current_positions`, sharing the same
                    #   boustrophedon convention as `_shift`) trace a smooth near-diagonal band, while
                    #   its last 196 rows (the nearest-neighbor-reordered MISSING FOVs, section 12) map
                    #   to `_shift` indices scattered across the full range, since that reordering
                    #   doesn't follow `_shift`'s own boustrophedon order.
                    #   Deployed to `S:\Leonardo\LT060_sample_04\merfish\MERci\notebooks\tests\` to run
                    #   interactively on the microscope; every figure saves to `SAMPLE_DIR/figures/` for
                    #   the user's own review before trusting the `_added` file -- in particular steps 1/2's
                    #   own stated scale limitation above means the shift's correctness is better judged
                    #   from step 3 (does the current grid visibly sit off the corrected tissue) and
                    #   step 6 (does MISSING trace a plausible tissue-edge strip) than from steps 1/2 alone.
data/
  configs/
    hal/            # hal-config-{mic}.xml — HAL config templates (one per microscope)
      mosaic_helper/ # hand-crafted per-microscope 10x/60x mosaic-tool setup configs (HAL + shutter
                     #   XML, e.g. hal-config-mf3-10x-mosaic-405.xml, shutter-config-mf3-405.xml) —
                     #   a dedicated subfolder so they're never matched by the HAL-template
                     #   auto-detection glob in prepare_imaging/01 (hal-config-*.xml directly under
                     #   data/configs/hal/); not every microscope has these, copy_mosaic_helper_configs
                     #   returns an empty list rather than erroring when none exist
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

`lineage_tracing/merfish_multi_z/` is a separate, **10-notebook** variant for a
variable-z-per-FOV acquisition (imaging each FOV only as deep as its own tissue
needs, instead of one fixed depth for every FOV) — see its own
`notebooks/prepare_imaging/lineage_tracing/merfish_multi_z/README.md` and the
package-layout entry above for the full per-notebook breakdown; the prose walkthrough
below (01-07) describes the shared 8-notebook template only.

**01** (`prepare_imaging/<variant>/01_create_hal_config_and_shutters.ipynb`): defines the imaging sequence as a *frame table* (one row per camera frame, columns `color`, `channel`, `z`) using `get_frame_table`. Supports `scan_mode="interleaved"` (all colors per z-plane, AOTF) or `scan_mode="sequential"` (full z-sweep per color, boustrophedon, physical shutters). The objective's return to `bead_z` after the stack is controlled by `z_return_mode`: `"progressive"` (default) steps down with blank frames in increments of `return_step` (5 µm default); `"instant"` jumps straight back (the previous behaviour). A per-channel `POWER = {nm: power}` dict sets the HAL config's `<default_power>` list (channel-ordered via `power_dict_to_channel_list`) — the actual acquisition power. Every shutter `<event>`'s own `<power>` is always a fixed `POWER_DEFAULT` (1.000 by default), regardless of frame colour: it is a full-modulation flag relative to `<default_power>`, not an independent absolute power, so writing the same real per-colour intensity into both places double-applies the scaling on real hardware (a bug this notebook had for a while — introduced in commit `39b9b58`, fixed by reverting `create_shutter_file` to always write a fixed power). Auto-generates a compact colour name via `get_color_sequence_name` (underscore-joined tokens, e.g. `blkf5_488f2_560f25_650f25_750f25`). Sets `<filetype>` (`.zarr` default, or `.dax`/`.tiff`) and `<exposure_time>` in the HAL config. Also writes a **transit** HAL config/shutter (`get_transit_frame_table`, `N_TRANSIT_BLANK` blank frames at bead z) for the between-boundary transit FOVs. Finally, copies any hand-crafted **mosaic-helper** HAL/shutter configs available for `MICROSCOPE` (`copy_mosaic_helper_configs`, reading `MERci/data/configs/hal/mosaic_helper/`) into `settings/` alongside the configs above -- these are pre-made 10x/60x low/high-mag setup files for running the Steve mosaic tool, not generated by MERci; a microscope with none available just prints that they're missing, not an error.

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

**04** (`prepare_imaging/<variant>/04_create_dave_config.ipynb`): builds the Dave experiment recipe XML from `round_info.csv` (notebook 03). With a single boundary it uses the classic single-positions recipe (one `<loop>` per imaging round). With **multiple boundaries** it builds a **per-segment** Dave recipe (`create_dave_config(positions_dir=…)`: each boundary/transit segment is its own `<loop>` — "Cells Imaging - <segment>" / "Hyb NN Imaging - <segment>" — with its own movie, HAL config and positions file, in order; fluidics loops stay between rounds, named "Hyb NN Fluidics" by the hyb index of the round they precede). Every `<loop>` gets its own identically-named `<loop_variable>`, even when several rounds share one positions file: Dave's real `v2Generator` (`handleLoop`) resolves a loop strictly by `self.loop_variable_names.index(loop.attrib["name"])`, so a movie's `<variable_entry>` cannot reference a differently-named loop_variable declared elsewhere — a shared/deduped loop_variable name loads fine in MERci's own reader but makes real Dave raise `ValueError: '<loop name>' is not in list` (confirmed directly against the storm_control source). The Kilroy config for the microscope is resolved (via `find_kilroy_config`, falling back to MF2 when the microscope has no config) and passed to `create_dave_config` as the source of fluidic protocol names: every protocol written into the Dave recipe is resolved to — and required to exist as — a `<protocol>` in that Kilroy config, raising `ValueError` otherwise. Writes `SAMPLE_DIR/settings/dave-{mic}-{N}hybs-{SAMPLE_NAME}.xml`, named via `dave.dave_config_filename(microscope, n_hybs, sample_name)` — a single source of truth for this filename shared with notebook 05's annotation step below.

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
- `transfer_dest` — network path (e.g. a NAS) to copy completed round data to; `None` = disabled (e.g. when `data_dir` is itself already a NAS-mounted path — nothing left to transfer). Runs only during the fluidics window (see `transfer_min_time`).
- `transfer_min_time` — minimum seconds remaining in the fluidics window before starting a transfer
- `analysis_mode` — `"same_drive"` (default, mode B: analyse from `data_dir` — also the mode to use when `data_dir` is itself a NAS-mounted path, since there is only one location to read from) or `"mirror_drive"` (mode A: mirror `data_dir` → `analysis_source_dir` during fluidics and analyse from that second-drive copy). Analysis runs **continuously** in both modes (not only during fluidics).
- `analysis_source_dir` — second-drive mirror directory; **required** when `analysis_mode="mirror_drive"`
- `n_analysis_workers` — FOV process-pool size; `None` = `cpu_count − 2` (`config.resolved_n_workers`). Each worker holds one image stack (~200 MB) in RAM.
- `mosaic_ffc_enabled` — flat-field-corrected round mosaics (default `True`); see `analysis/ffc.py`. `ffc_fov_selection_strategy` (`"exterior_grid"` default / `"emptiest_stats"` / `"single_fov_all_frames"`), `ffc_connectivity`, `ffc_neighbor_tolerance`, `ffc_smooth_sigma_px`, `ffc_normalize_percentile`, `ffc_min_value`, `ffc_min_samples`, `ffc_emptiest_n_fovs`, and `mosaic_contrast_percentile_clip` (the one shared whole-canvas contrast stretch) tune it.

`ExperimentMetadata` (loaded via `ExperimentMetadata.load(round_info_csv, positions_txt, data_dir)`) cross-references round IDs, FOV IDs, series patterns, and expected file paths. When a `dir`/`data_dir` column is present in `round_info.csv`, per-round file paths are resolved from that directory instead of the top-level `data_dir` — and that column's value can be **either** an absolute path **or** a path relative to *this machine's own* `data_dir`'s parent (e.g. `data/hybs/H01` — for a manually relocated experiment folder). Each series carries an ordered list of **candidate directories** (`SeriesInfo.candidate_dirs`); `resolve_path(fov, suffix)` returns the first candidate that exists on disk. If none of the exact-width-guessed candidates exist, it falls back to a **width-agnostic directory scan** (`SeriesInfo._scan_dir_for_fov`, cached per directory so 1000+ per-FOV lookups against one round only pay for one scan): `fov_from_stem` was already built from `\d+` (never a fixed digit count — see `_pattern_to_regex`), just previously only used in the reverse (path → FOV id) direction; this reuses it forward too, so a `round_info.csv` whose `series` pattern's zero-pad width has gone stale (e.g. positions.txt regenerated with more FOVs, needing more digits, after round_info.csv was already written) still finds the real files instead of silently reporting them all missing. Only once that scan also comes up empty does it fall back to the primary candidate (e.g. before acquisition, when the FOV genuinely doesn't exist yet). The **cells round** is treated as a bona fide imaging round (typically `imaging_round=1`) and its files are accepted in **either** `data/cells/` or the top-level `data/`, regardless of which the `data_dir` column records — so `all_fovs_done_for_round`, mosaics, and transfers all find the cells data wherever HAL actually wrote it.

`ExperimentStateMonitor` determines the microscope phase by watching the newest file mtime in `data_dir`:
- **IMAGING**: a new image file was written within `imaging_idle_threshold` seconds
- **FLUIDICS**: `t_min ≤ time_since_imaging ≤ t_max` → `should_analyze = True`

`should_analyze` is no longer the analysis gate — FOV/round analysis runs continuously. The phase is still used to time the mode-A mirror and the NAS transfer (both read the acquisition drive, so both run only while the microscope is idle).

`ProgressTracker` tracks completeness via zero-byte sentinel files under `analysis_dir/done/`:
- `<stem>.fov_done` — FOV analysis complete
- `round_<r>.round_done` — mosaic(s) built for round r
- `round_<r>.round_transferred` — raw data for round r copied to `transfer_dest`
- `round_<r>.fov_submitted` / `round_<r>.round_mosaic_submitted` — cluster-side SLURM submission bookkeeping (hold the submitted job id as text, not just an empty touch — see "Moving QC analysis to a SLURM cluster" below)
- `ffc_<color>nm.ffc_done` — flat-field-correction field computed and cached for that color (once per experiment, not per round — see `analysis/ffc.py`)

Multiple notebooks can run concurrently — no shared state.

`FOVScheduler.run_loop()` runs **continuously** (acquisition + fluidics): each tick it (in mirror mode) refreshes the second-drive mirror while idle, discovers stable image files (zarr/dax/tiff) under `config.analysis_data_dir`, and analyses pending files **in parallel across a process pool** (`config.resolved_n_workers`). Each worker runs the top-level `analysis.fov.analyze_file`, which reads the stack once and writes thumbnails (PNG) + per-frame stats (CSV) + histograms (`.npz`) + the FOV sentinel. With `n_analysis_workers=1` it runs serially in-process. Respects `fov_subset`; call `.close()` (done automatically when `run_loop` exits) to shut the pool down. `RoundScheduler.run_loop()` also runs continuously, assembling **one mosaic per imaging color** (`round_{r:03d}_{color}nm_mosaic.png`) once all FOV sentinels exist; auto-resolves `flip_y` from the HAL config; optional background transfers via `transfer.transfer_round` happen only during the fluidics window. When `config.mosaic_ffc_enabled` (default `True`), `build_round_mosaics` builds flat-field-corrected mosaics instead: it ensures that color's FFC field is cached (`analysis.ffc.compute_and_cache_ffc` — a no-op after the first round, since the field is computed once per experiment per color and reused), reads every FOV's raw frame directly (`analysis.round.load_raw_frames_for_round`), and assembles the mosaic via `analysis.round.create_mosaic_ffc` (divides out the FFC field, crops each FOV's overlap border, applies one shared contrast stretch across the whole canvas) instead of `create_mosaic`'s independent-per-tile-contrast pre-made thumbnails. See `analysis/ffc.py`'s package-layout entry above for the full design. `ExperimentScheduler.wait_and_run()` calls a user callback after all rounds complete. `FOVScheduler._build_task`/`RoundScheduler._analyse_one_round`'s path/kwarg-construction logic is factored out into module-level `build_fov_task_kwargs`/`build_round_mosaics`/`resolve_round_flip_y`/`resolve_round_color_frame_indices`/`source_dirs_for_round` functions in `scheduler.py` — both the schedulers and the cluster-side CLI scripts below call these, so local and cluster analysis can never disagree about where outputs land.

**Moving QC analysis to a SLURM cluster.** `01_fov_scheduler.ipynb`/`02_round_scheduler.ipynb` remain fully supported for running QC locally on the microscope computer, but analysis can instead run entirely on a SLURM cluster, freeing the microscope computer:

- **`07_cluster_submit_analysis.ipynb`** (run on a cluster login/transfer node, after you've moved data from the microscope/NAS to cluster storage yourself — e.g. via Globus/FileZilla; no automation for that leg exists in this repo) — builds `ExperimentConfig`/`ExperimentMetadata`/`ProgressTracker` pointed at the cluster-side `SAMPLE_DIR` exactly like `05_batch_sample_review.ipynb` does, then for each round with pending FOVs (`tracker.pending_fov_files`, which only ever returns files that already exist — partial/incremental arrival is fine) writes a manifest and submits a SLURM **array job** (one task per FOV) via `acquisition/cluster_submit.py`, and similarly submits a mosaic-building job once a round's FOVs are all done (`tracker.pending_rounds`). Submission is tracked with new `ProgressTracker` sentinels — `round_<r>.fov_submitted` / `round_<r>.round_mosaic_submitted` (holding the submitted SLURM job id as text, not just an empty touch) — checked against `cluster_submit.is_job_active` (an `sacct`-based query) before resubmitting, so re-running the notebook while a previous array job is still `PENDING`/`RUNNING` is a no-op.
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
`notebooks/misc/measure_tissue_thickness_test.ipynb` for the reference
implementation.

### Diagnostic images and visual verification

This project's diagnostic/QC notebooks (e.g. `misc/correct_camera_rotation.ipynb`,
mosaic viewers) live and die by image-based verification, so the global
visual-verification protocol (`~/.claude/CLAUDE.md`) applies, plus these
repo-specific habits learned from a real, multi-hour miscommunication during
the camera-rotation-correction work (see
`prompt_history/2026_07_31_1932_confirm_camera_rotation_orientation.md` and
`..._1939_fix_camera_rotation_mosaic_never_oriented.md`):

- Save every diagnostic image meant for the user's own eyes to a real path
  under the experiment's own tree (e.g. `SAMPLE_DIR/analysis/figures/`) or the
  repo, never only the session scratchpad — and state the literal path.
- When a notebook section is redesigned/replaced, delete or rename the old
  diagnostic PNGs it produced instead of leaving them in place under a
  similar filename. A stale image with no version marker is indistinguishable
  from the current one and caused a real, hours-long false alarm (the user
  was comparing a pre-fix `mosaic_comparison.png` while the notebook itself
  had already been fixed).
- Before telling the user a diagnostic output is "already correct" based on a
  rendered image, confirm the code path that produced that image actually
  applies every transform being claimed (e.g. grep for
  `apply_microscope_orientation` or the equivalent) — a plausible-looking
  picture is not proof the code that built it is right, particularly for
  tissue images whose texture may not show a raw-vs-corrected mismatch
  clearly at a glance.

## Running notebooks

Notebooks auto-detect `SAMPLE_DIR` from their own location. `analysis/`, `during_imaging/`, and `misc/` notebooks are two levels under the repo root, so `MERCI_DIR = Path(os.getcwd()).parent.parent` (the `MERci/` clone), then `SAMPLE_DIR = MERCI_DIR.parent`. The `prepare_imaging/<variant>/` notebooks (`reference`) are **three** levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent`; the `tumor/{epi,disk}/` and `lineage_tracing/{merfish,lineage}/` notebooks are **four** levels deep, so they use `MERCI_DIR = Path(os.getcwd()).parent.parent.parent.parent`. Do not hardcode absolute paths in notebooks.

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

## Working / cache files

Any working/intermediate file Claude generates for this project — notebook-
generator scripts used to build an executed notebook via `nbclient`, a
diagnostic image meant for the user to inspect, a migration backup, the
internal verbatim-capture buffer (below) — goes under `cache/` (repo root,
gitignored), never the session scratchpad or any location outside this repo.
Structure inside `cache/` is flexible; when pointing the user at a specific
file, always give the literal path under `cache/`. This is distinct from
`analysis/cache/<notebook_name>/` (see "Notebook coding guidelines" above),
which is a per-*experiment* cache for a notebook's own calculation results,
living under `SAMPLE_DIR/analysis/`, not this repo.

## Remembering task history

This project keeps two complementary, local-only records, both gitignored:

1. **`prompt_history/`** — the log. One file per request: frontmatter, then
   `## Prompt` / `## Plan` / `## Summary`, plus `## Learning` and
   `## Verbatim History` when applicable (all described below). This is now
   the single place to look for what happened on any given request —
   including Claude's own verbatim turn-by-turn text, folded directly into
   each entry.
2. **`FINDINGS.md`** — *current state*. Curated, deduplicated head: what is
   true now, what was wrong, and the open next step. Read this first when
   resuming.

**Maintenance habit:** for **every user question/request**, log it to
`prompt_history/` (below). When that entry changes a conclusion or project
state, also update the relevant `FINDINGS.md` section. `prompt_history/` is
no longer strictly append-only as a *file* (a past entry's structure may be
fixed up if it doesn't fit the current template) — but the **`## Prompt`
text itself must always stay verbatim**: never rephrase, summarize, or
paraphrase it, whether writing a new entry or fixing up an old one.
`prompt_history/` itself has **two methods** depending on how the prompt
arrives:

### Method 1 — user pre-writes the prompt as a file

The user creates a file in `prompt_history/` whose name is **only the date/time**
(e.g. `2026_06_18_1002.txt`) and writes their request inside it. When asked to read
and act on it:

0. **If more than one such date/time-only `.txt` file is currently pending**
   (e.g. `2026_08_01_1844.txt`, `2026_08_01_1850.txt`, `2026_08_01_1851.txt` all
   sitting unprocessed at once), always start with the one written **first**
   (earliest timestamp in the filename) — in that example, `2026_08_01_1844.txt`.
1. Read the file and carry out the request.
2. Rewrite the file to start with the YAML frontmatter, followed by `## Prompt`
   (the file's original raw text, moved under this heading verbatim — never
   rephrased), then `## Plan` / `## Summary` and the rest of the shared format
   below.
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
<verbatim copy of the user's request -- never rephrased>

## Plan
<Claude's plan of action before executing>

## Summary
<what was actually done, including any deviations from the plan>

## Learning
<optional -- only when the Summary above already surfaced a concrete, generalizable
lesson/rule/gotcha useful beyond this one request (e.g. a tool quirk, a wrong
assumption corrected, a verification habit that mattered). Omit this heading
entirely when nothing like that came up -- do not invent one to fill the slot.>

## Verbatim History
<Claude's own verbatim turn-by-turn text for this request, folded in from the
internal cache/verbatim_buffer/ (below) when the entry is finalized. Omit this
heading if no buffer content exists for this request (e.g. very old entries
predating the buffer).>
```

The `UserPromptSubmit` date/time hook injects `Current local date/time: … (epoch N)`
on every prompt. Compute **`elapsed`** (written just before `status`) as the finish
time minus that submit epoch: run `date +%s` (bash) or
`[DateTimeOffset]::Now.ToUnixTimeSeconds()` (PowerShell) when done and subtract the
epoch from the message that began the request (the first turn's epoch for a
multi-turn request). Omit `elapsed` if no submit epoch is available — never guess it.

Format rationale: Markdown + YAML frontmatter is Claude-native, human-readable,
and lets all entries be scanned/grepped by metadata without reading every body.

### Verbatim capture (internal mechanism, not a separate tier to read)

A `Stop` hook (`.claude/hooks/save_verbatim.ps1`, wired into `.claude/settings.json`)
still runs automatically after every turn and appends Claude's exact assistant text
to a per-day buffer, `cache/verbatim_buffer/{YYYY-MM-DD}_verbatim.md` — no action
needed from Claude for the capture itself. This buffer is purely an internal
staging area; never point the user at it. When finalizing a `prompt_history/`
entry, fold that request's slice of the current day's buffer into the entry's own
`## Verbatim History` section (matching by timestamp — the slice from this
request's own start up to the next request's start), then clear/truncate that
portion of the buffer so it isn't re-ported into a later entry.

### FINDINGS.md — current state

`FINDINGS.md` (repo root, gitignored) is the curated, deduplicated head of
what's true about this project right now — not a log. Read it first when
resuming work after a gap; update it whenever a `prompt_history/` entry
changes a conclusion, fixes something that was wrong, or completes an open
next step. Keep it short and current rather than exhaustive — `prompt_history/`
is the append-only provenance trail; `FINDINGS.md` is just the live summary.

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
(gitignored, personal-only, like `prompt_history/`) — a
narrative walkthrough with real code snippets, dead ends, and the moment a
hypothesis got confirmed or falsified. Not needed for mechanical tasks. See
`~/.claude/commands/rationale.md` (the `/rationale` command) for the exact
template/style, and offer one proactively at the end of an investigation-
heavy task rather than waiting to be asked — the reasoning is easiest to
capture while still fresh in context.

