window.BENCHMARK_DATA = {
  "lastUpdate": 1794918968934,
  "repoUrl": "https://github.com/serapeum-org/pyramids-core",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "name": "Mostafa Farrag",
            "username": "MAfarrag",
            "email": "moah.farag@gmail.com"
          },
          "committer": {
            "name": "Mostafa Farrag",
            "username": "MAfarrag",
            "email": "moah.farag@gmail.com"
          },
          "id": "cada522c9308b402107cdb4952b7e3b3d6aeced0",
          "message": "docs(benchmarks): reflow the track-step comment and mark the abbreviated command\n\nRound-2 review nits, comment/doc only (no runtime change; YAML still parses as a\n5-step job with permissions: contents: write).\n\n- N1: the L3 edit left \"When a benchmark\" dangling at a line end, splitting the\n  regression sentence mid-clause. Reflow so it starts on its own comment line.\n- N2: the `benchmark` pixi task is actually\n  `pytest benchmarks --benchmark-only --benchmark-columns=min,median,mean,stddev,rounds`,\n  so the shorthand `pytest benchmarks --benchmark-only` in benchmarks.yml and\n  docs/how-to/testing.md is not the literal command. Append \"…\" to mark it\n  abbreviated (the omitted flag is display-only column formatting).\n\nCorrection to commit 5abd3ef367 (round-1 M1), whose embedded grep proof\nunder-counted: the real pin tally is 8 occurrences of\n961524686fb512c05ca8e4c7bfe81ba68eef4dda across 6 files — benchmarks.yml (1),\ntests.yml (2), lint.yml (2), pip-audit.yml (1), github-pages-mkdocs.yml (1),\npure-wheel-test.yml (1) — i.e. 7 others + benchmarks, not \"5 → 6\". The shipped\npin is correct and uniform; recorded here (the pushed message is not amended, per\nthe no-force-push rule) so a future repo-wide SHA sweep covers all 6 files.",
          "timestamp": "2026-11-16T13:55:41Z",
          "url": "https://github.com/serapeum-org/pyramids-core/commit/cada522c9308b402107cdb4952b7e3b3d6aeced0"
        },
        "date": 1794839038403,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_collection.py::test_from_dataset",
            "value": 2819.78645572845,
            "unit": "iter/sec",
            "range": "stddev: 0.000028787186826392597",
            "extra": "mean: 354.63678391974713 usec\nrounds: 398"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_mean_reduction",
            "value": 27.534738128506127,
            "unit": "iter/sec",
            "range": "stddev: 0.004129556760007528",
            "extra": "mean: 36.31775959999857 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_apply",
            "value": 36.384354958462986,
            "unit": "iter/sec",
            "range": "stddev: 0.0009063000327696618",
            "extra": "mean: 27.48434048484898 msec\nrounds: 33"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_to_cog_stack",
            "value": 2.941500442256067,
            "unit": "iter/sec",
            "range": "stddev: 0.00159312915501557",
            "extra": "mean: 339.96255299999945 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_merge",
            "value": 4.820890664023946,
            "unit": "iter/sec",
            "range": "stddev: 0.003087841494947013",
            "extra": "mean: 207.4305496000008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_fishnet_build",
            "value": 102.35551338957647,
            "unit": "iter/sec",
            "range": "stddev: 0.0006722860308747157",
            "extra": "mean: 9.769869417721434 msec\nrounds: 79"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_rasterize",
            "value": 2.3895308022217203,
            "unit": "iter/sec",
            "range": "stddev: 0.06858377889715919",
            "extra": "mean: 418.49219899999923 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_with_centroid",
            "value": 2.5797494589633394,
            "unit": "iter/sec",
            "range": "stddev: 0.06818299840913666",
            "extra": "mean: 387.63454199999927 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_geodesic_area",
            "value": 1.543422259544326,
            "unit": "iter/sec",
            "range": "stddev: 0.003536828756263049",
            "extra": "mean: 647.9108318000002 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_memory.py::test_lazy_read_is_memory_bounded",
            "value": 8.489563148021299,
            "unit": "iter/sec",
            "range": "stddev: 0.0038035992621062815",
            "extra": "mean: 117.79169111111149 msec\nrounds: 9"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_read_array_eager",
            "value": 90.53103164444214,
            "unit": "iter/sec",
            "range": "stddev: 0.0008886536089689364",
            "extra": "mean: 11.045936203703825 msec\nrounds: 54"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_read_array_lazy",
            "value": 39.784328220613766,
            "unit": "iter/sec",
            "range": "stddev: 0.0015669228432805989",
            "extra": "mean: 25.135525588235573 msec\nrounds: 17"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_isel",
            "value": 64.3127644641347,
            "unit": "iter/sec",
            "range": "stddev: 0.0014682866418520282",
            "extra": "mean: 15.54901283333373 msec\nrounds: 66"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_sel",
            "value": 57.01082466869508,
            "unit": "iter/sec",
            "range": "stddev: 0.0006895827965321245",
            "extra": "mean: 17.540528589285692 msec\nrounds: 56"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_subset",
            "value": 344.3111509505324,
            "unit": "iter/sec",
            "range": "stddev: 0.0002378688589063922",
            "extra": "mean: 2.9043497349398106 msec\nrounds: 249"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_from_array",
            "value": 178.94939783494988,
            "unit": "iter/sec",
            "range": "stddev: 0.00043781906214401025",
            "extra": "mean: 5.5881719195407875 msec\nrounds: 87"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_array_full",
            "value": 779.9061478143765,
            "unit": "iter/sec",
            "range": "stddev: 0.00023573195194758932",
            "extra": "mean: 1.2822055612748002 msec\nrounds: 408"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_array_bbox",
            "value": 455.75600778943056,
            "unit": "iter/sec",
            "range": "stddev: 0.00011276776328366482",
            "extra": "mean: 2.1941564848488455 msec\nrounds: 231"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_array_decimated",
            "value": 6242.987477608808,
            "unit": "iter/sec",
            "range": "stddev: 0.000011584893563593747",
            "extra": "mean: 160.17972222219169 usec\nrounds: 1116"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_file_from_disk",
            "value": 16.42898264852165,
            "unit": "iter/sec",
            "range": "stddev: 0.0005472216274306205",
            "extra": "mean: 60.86804164285756 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_write_geotiff",
            "value": 5.050416970042846,
            "unit": "iter/sec",
            "range": "stddev: 0.0017351419278628022",
            "extra": "mean: 198.00345316666323 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_to_cog",
            "value": 3.2885106027741124,
            "unit": "iter/sec",
            "range": "stddev: 0.0046109729626252665",
            "extra": "mean: 304.0890302000008 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_full_read",
            "value": 20.628612810522135,
            "unit": "iter/sec",
            "range": "stddev: 0.00028873813136237055",
            "extra": "mean: 48.47635704762102 msec\nrounds: 21"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_read_part_decimated",
            "value": 270.2101961566252,
            "unit": "iter/sec",
            "range": "stddev: 0.00007608879852272497",
            "extra": "mean: 3.70082259745801 msec\nrounds: 236"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_preview",
            "value": 273.5740999167368,
            "unit": "iter/sec",
            "range": "stddev: 0.00009993081499638714",
            "extra": "mean: 3.655316787314126 msec\nrounds: 268"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_point",
            "value": 299.25127840808494,
            "unit": "iter/sec",
            "range": "stddev: 0.000057146844662842685",
            "extra": "mean: 3.3416732764506802 msec\nrounds: 293"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_read_tile",
            "value": 80.33747818943554,
            "unit": "iter/sec",
            "range": "stddev: 0.00028383509222498597",
            "extra": "mean: 12.447490542857256 msec\nrounds: 70"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_crop_bbox",
            "value": 42.6568247782384,
            "unit": "iter/sec",
            "range": "stddev: 0.0013721590652798316",
            "extra": "mean: 23.442907558139567 msec\nrounds: 43"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_reproject_to_web_mercator",
            "value": 619.19662709684,
            "unit": "iter/sec",
            "range": "stddev: 0.00004249409931423593",
            "extra": "mean: 1.6149958773008688 msec\nrounds: 163"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_resample_coarsen",
            "value": 14.843217979971401,
            "unit": "iter/sec",
            "range": "stddev: 0.00026988526123955417",
            "extra": "mean: 67.37083571428671 msec\nrounds: 14"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_align_to_grid",
            "value": 7.020261563011522,
            "unit": "iter/sec",
            "range": "stddev: 0.0008127742143103301",
            "extra": "mean: 142.4448349999974 msec\nrounds: 8"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_warped_view_windowed_read",
            "value": 81.89962057706289,
            "unit": "iter/sec",
            "range": "stddev: 0.0005148112299060328",
            "extra": "mean: 12.210068776314499 msec\nrounds: 76"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_to_polygons",
            "value": 1.329044646252574,
            "unit": "iter/sec",
            "range": "stddev: 0.0028614344083895587",
            "extra": "mean: 752.4201710000028 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_contour",
            "value": 42.7547957726117,
            "unit": "iter/sec",
            "range": "stddev: 0.0004187942966926203",
            "extra": "mean: 23.38918902380982 msec\nrounds: 42"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_sieve",
            "value": 234.55498403159137,
            "unit": "iter/sec",
            "range": "stddev: 0.0002554691541022062",
            "extra": "mean: 4.263392671567847 msec\nrounds: 204"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_proximity",
            "value": 271.8321555551559,
            "unit": "iter/sec",
            "range": "stddev: 0.00003595793710909527",
            "extra": "mean: 3.678740647726997 msec\nrounds: 264"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Mostafa Farrag",
            "username": "MAfarrag",
            "email": "moah.farag@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c57dd67928040631f4793e3a0b7206c5c0b4e6a7",
          "message": "docs(netcdf): replace the chunks= dimension-name keys no backend accepts (#1584)\n\n  Twelve snippets across six files documented `chunks=` as a mapping keyed                                   \n  by dimension name (`{\"time\": 24, \"lat\": 256, \"lon\": 256}`). No installed                                   \n  backend accepts that spelling: a chunks dict is keyed by integer axis                                      \n  index, or by the case-insensitive aliases `rows` / `cols` / `columns` /                                    \n  `bands`. Nine of the twelve raised `ValueError`; the other three sit on                                    \n  the `animate=` path, where the argument is discarded before the renderer                                   \n  sees it, so they silently did nothing at all.                                                              \n                                                                                                             \n  The spelling was not invented. `NetCDF.chunk(...).chunks` reports chunk                                    \n  sizes keyed by dimension name, so the read side cannot consume the write                                   \n  side's output and the pages used the spelling the API itself advertises.                                   \n  Teaching `chunks=` to accept dimension names is a backend decision and                                     \n  is not attempted here.                                                                                     \n                                                                                                             \n  - Rewrite all twelve: five in the lazy tutorial, two in the plotting                                       \n    tutorial, one each in the netcdf reference index and plot pages, one                                     \n    in README, and two code cells in the dask-lazy notebook that a                                           \n    quote-sensitive grep had missed, since .ipynb escapes its quotes.                                        \n  - State the accepted keys, their edges and the round-trip asymmetry at                                     \n    every seam documenting the argument: `read_array`, `plot`, `to_xarray`                                   \n    (facade and engine), `chunk`, `open_mfdataset` and the mesh                                              \n    `open_mfdataset`.                                                                                        \n  - Correct three seam behaviours documented wrongly or not at all:                                          \n    `to_xarray` swallows an unusable key and falls back to an eager read                                     \n    per variable instead of raising; `open_mfdataset` resolves keys                                          \n    against the per-file variable and treats `chunks=None` as `\"auto\"`                                       \n    rather than as the eager path every other seam gives it; and keys                                        \n    resolve against the stored rank, which `VariableInfo.shape` need not                                     \n    show, so `{0: 1}` can silently chunk rows.                                                               \n  - Add `tests/netcdf/io/test_chunks_key_contract.py` and four plot cases                                    \n    — 52 tests pinning the accepted vocabulary by resolved chunk layout                                      \n    rather than by \"did not raise\", every rejection class, `bands` arity                                     \n    across 2-D/3-D/4-D, the non-round-trippable `chunk()` report, the                                        \n    per-variable `to_xarray` fallback, and the animate-inert versus                                          \n    static-raises asymmetry.\n                                                                                                           \n  Nothing in these pages had ever been executed: no markdown-exec,                                           \n  pytest-codeblocks or mktestdocs is configured, and nbval skips the one                                     \n  notebook involved. Every spec written here was run against a variable of                                   \n  the shape its page describes. No executable source line changed — the                                      \n  three touched modules are AST-identical to main once docstrings are                                        \n  stripped.                                                                                                  \n                                                                                                             \n  Closes #1582",
          "timestamp": "2026-11-16T16:11:41Z",
          "url": "https://github.com/serapeum-org/pyramids-core/commit/c57dd67928040631f4793e3a0b7206c5c0b4e6a7"
        },
        "date": 1794918949799,
        "tool": "pytest",
        "benches": [
          {
            "name": "benchmarks/test_bench_collection.py::test_from_dataset",
            "value": 5026.0587481725115,
            "unit": "iter/sec",
            "range": "stddev: 0.00003561879928749374",
            "extra": "mean: 198.9630543740864 usec\nrounds: 423"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_mean_reduction",
            "value": 37.316399768100894,
            "unit": "iter/sec",
            "range": "stddev: 0.0006067570923932161",
            "extra": "mean: 26.79786919998719 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_apply",
            "value": 39.052644219669276,
            "unit": "iter/sec",
            "range": "stddev: 0.00013834466378714853",
            "extra": "mean: 25.60646071428729 msec\nrounds: 35"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_to_cog_stack",
            "value": 3.96992252669196,
            "unit": "iter/sec",
            "range": "stddev: 0.0004989150961447719",
            "extra": "mean: 251.8940843999985 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_collection.py::test_merge",
            "value": 6.668647477255602,
            "unit": "iter/sec",
            "range": "stddev: 0.00037914210528703187",
            "extra": "mean: 149.95544500000133 msec\nrounds: 7"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_fishnet_build",
            "value": 139.21184849687316,
            "unit": "iter/sec",
            "range": "stddev: 0.00025917814286865527",
            "extra": "mean: 7.183296614457792 msec\nrounds: 83"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_rasterize",
            "value": 4.262888871358902,
            "unit": "iter/sec",
            "range": "stddev: 0.05201104016920388",
            "extra": "mean: 234.58270439999183 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_with_centroid",
            "value": 4.172433160531022,
            "unit": "iter/sec",
            "range": "stddev: 0.05375852185481761",
            "extra": "mean: 239.66830900000105 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_feature.py::test_geodesic_area",
            "value": 3.066018530135452,
            "unit": "iter/sec",
            "range": "stddev: 0.000519701483091169",
            "extra": "mean: 326.1558891999982 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_memory.py::test_lazy_read_is_memory_bounded",
            "value": 10.607785899608832,
            "unit": "iter/sec",
            "range": "stddev: 0.003003465513183121",
            "extra": "mean: 94.27037927272605 msec\nrounds: 11"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_read_array_eager",
            "value": 129.24376213625288,
            "unit": "iter/sec",
            "range": "stddev: 0.00008499300105991292",
            "extra": "mean: 7.737317325580232 msec\nrounds: 86"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_read_array_lazy",
            "value": 52.83213456607938,
            "unit": "iter/sec",
            "range": "stddev: 0.0011260535071480992",
            "extra": "mean: 18.92787426086784 msec\nrounds: 23"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_isel",
            "value": 114.13139366073554,
            "unit": "iter/sec",
            "range": "stddev: 0.00008492233308966887",
            "extra": "mean: 8.761831148514474 msec\nrounds: 101"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_sel",
            "value": 95.21933860344716,
            "unit": "iter/sec",
            "range": "stddev: 0.0001587459515651797",
            "extra": "mean: 10.502068326315781 msec\nrounds: 95"
          },
          {
            "name": "benchmarks/test_bench_netcdf.py::test_subset",
            "value": 380.89259895186217,
            "unit": "iter/sec",
            "range": "stddev: 0.00004760226231658306",
            "extra": "mean: 2.625412000001558 msec\nrounds: 317"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_from_array",
            "value": 243.75417010051828,
            "unit": "iter/sec",
            "range": "stddev: 0.00016703402904399498",
            "extra": "mean: 4.102493916668685 msec\nrounds: 108"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_array_full",
            "value": 844.6131887968054,
            "unit": "iter/sec",
            "range": "stddev: 0.000011315949554415358",
            "extra": "mean: 1.1839739341799187 msec\nrounds: 471"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_array_bbox",
            "value": 541.569601039206,
            "unit": "iter/sec",
            "range": "stddev: 0.00012069019524611471",
            "extra": "mean: 1.8464847326753977 msec\nrounds: 303"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_array_decimated",
            "value": 4480.302173586463,
            "unit": "iter/sec",
            "range": "stddev: 0.00001826519077647777",
            "extra": "mean: 223.19923104639707 usec\nrounds: 1108"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_read_file_from_disk",
            "value": 18.92381084620482,
            "unit": "iter/sec",
            "range": "stddev: 0.00012500889597044518",
            "extra": "mean: 52.843478944440555 msec\nrounds: 18"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_write_geotiff",
            "value": 5.798074707933415,
            "unit": "iter/sec",
            "range": "stddev: 0.0006321742235112594",
            "extra": "mean: 172.4710443333398 msec\nrounds: 6"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_to_cog",
            "value": 4.151130707148982,
            "unit": "iter/sec",
            "range": "stddev: 0.01654987571863747",
            "extra": "mean: 240.898220400004 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_full_read",
            "value": 26.501036289162695,
            "unit": "iter/sec",
            "range": "stddev: 0.00015582558849561024",
            "extra": "mean: 37.73437344444296 msec\nrounds: 27"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_read_part_decimated",
            "value": 355.89657524330715,
            "unit": "iter/sec",
            "range": "stddev: 0.00004320056538483789",
            "extra": "mean: 2.8098050657451656 msec\nrounds: 289"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_preview",
            "value": 360.27007517965905,
            "unit": "iter/sec",
            "range": "stddev: 0.00002970233371633724",
            "extra": "mean: 2.775695426552764 msec\nrounds: 354"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_point",
            "value": 388.24484507543946,
            "unit": "iter/sec",
            "range": "stddev: 0.00002416924217148037",
            "extra": "mean: 2.5756942112282033 msec\nrounds: 374"
          },
          {
            "name": "benchmarks/test_bench_raster_io.py::test_cog_read_tile",
            "value": 106.31690265656145,
            "unit": "iter/sec",
            "range": "stddev: 0.00019960041237306026",
            "extra": "mean: 9.405842109888479 msec\nrounds: 91"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_crop_bbox",
            "value": 49.58781096343339,
            "unit": "iter/sec",
            "range": "stddev: 0.0004851813848284136",
            "extra": "mean: 20.166246111114106 msec\nrounds: 45"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_reproject_to_web_mercator",
            "value": 939.3765069793117,
            "unit": "iter/sec",
            "range": "stddev: 0.000023752403664208304",
            "extra": "mean: 1.0645358837167764 msec\nrounds: 215"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_resample_coarsen",
            "value": 19.91542726606348,
            "unit": "iter/sec",
            "range": "stddev: 0.0009735615812075036",
            "extra": "mean: 50.212329700002556 msec\nrounds: 20"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_align_to_grid",
            "value": 9.725533747552529,
            "unit": "iter/sec",
            "range": "stddev: 0.001215762728577843",
            "extra": "mean: 102.82212019999974 msec\nrounds: 10"
          },
          {
            "name": "benchmarks/test_bench_spatial.py::test_warped_view_windowed_read",
            "value": 105.14747959562338,
            "unit": "iter/sec",
            "range": "stddev: 0.0007209467414413704",
            "extra": "mean: 9.51045145205386 msec\nrounds: 73"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_to_polygons",
            "value": 1.9728592186888854,
            "unit": "iter/sec",
            "range": "stddev: 0.0009433985491211126",
            "extra": "mean: 506.87853980000455 msec\nrounds: 5"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_contour",
            "value": 62.54689988073615,
            "unit": "iter/sec",
            "range": "stddev: 0.00024078577361970197",
            "extra": "mean: 15.988002633332599 msec\nrounds: 60"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_sieve",
            "value": 283.43596605996913,
            "unit": "iter/sec",
            "range": "stddev: 0.000025093074912068057",
            "extra": "mean: 3.52813375769122 msec\nrounds: 260"
          },
          {
            "name": "benchmarks/test_bench_vectorize.py::test_proximity",
            "value": 391.4193877409878,
            "unit": "iter/sec",
            "range": "stddev: 0.00007206660282798692",
            "extra": "mean: 2.554804466307442 msec\nrounds: 371"
          }
        ]
      }
    ]
  }
}