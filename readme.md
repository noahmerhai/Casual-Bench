# 1. Install dependencies
pip install dowhy networkx pandas anthropic
pip install git+https://github.com/noahmerhai/ananke-py3.13.git

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Put the files in your repo
cd ADMG-gen
mkdir causal_bench
# move all 5 python files into causal_bench/

# 4. Dry run first (no API calls, just see prompts + ground truth)
cd causal_bench
python benchmark_runner.py --results ../stress_test_results.json --dry-run --max-nodes 8 --task full

# 5. Run for real on small graphs
python benchmark_runner.py --results ../stress_test_results.json --max-nodes 8 --task full

# 6. Run just one task type if you want
python benchmark_runner.py --results ../stress_test_results.json --max-nodes 8 --task id
python benchmark_runner.py --results ../stress_test_results.json --max-nodes 8 --task backdoor
python benchmark_runner.py --results ../stress_test_results.json --max-nodes 8 --task frontdoor

# 7. Run on all 100 graphs (will take longer, more API calls)
python benchmark_runner.py --results ../stress_test_results.json --task combined