window.BENCHMARK_DATA = {
  "lastUpdate": 1794839060672,
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
      }
    ]
  }
}