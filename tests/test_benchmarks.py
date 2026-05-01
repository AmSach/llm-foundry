from llm_foundry import BenchmarkSuite, FixedBackend, default_benchmark_cases


def test_benchmark_suite_runs():
    suite = BenchmarkSuite(FixedBackend("blue"))
    report = suite.run(default_benchmark_cases())
    assert report.total == 4
    assert report.passed >= 1
    assert report.mean_latency_ms >= 0
