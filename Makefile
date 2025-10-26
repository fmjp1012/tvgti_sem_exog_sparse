PYTHON ?= /Users/fmjp/venv/default/bin/python
NUM_TRIALS ?= 100
TUNE_ARGS ?=
RUN_ARGS ?=
HYPER ?=

.PHONY: help \
        tune_piecewise tune_linear \
        run_piecewise run_linear \
        tune_run_piecewise tune_run_linear

help:
	echo "利用可能なターゲット:"
	@echo "  make tune_piecewise   # ピースワイズのハイパラ調整のみ (TUNE_ARGSで追加指定)"
	@echo "  make tune_linear      # 線形のハイパラ調整のみ"
	@echo "  make run_piecewise    # ハイパラJSONを使ってピースワイズを実行 (HYPERでパス指定)"
	@echo "  make run_linear       # ハイパラJSONを使って線形を実行"
	@echo "  make tune_run_piecewise  # 調整→100試行実行を一括実行"
	@echo "  make tune_run_linear     # 調整→100試行実行を一括実行"
	@echo "共通変数: NUM_TRIALS, TUNE_ARGS, RUN_ARGS, HYPER"
	@echo "  TUNE_ARGS で探索範囲も上書き可 (例: --methods pp,pc --pp_rho_low 1e-5 --pc_C_choices 1,3,5)"

tune_piecewise:
	$(PYTHON) -m code.tune_and_run piecewise --no_run $(TUNE_ARGS)
	

tune_linear:
	$(PYTHON) -m code.tune_and_run linear --no_run $(TUNE_ARGS)

run_piecewise:
	@CMD=""; \
	if [ -n "$(HYPER)" ]; then \
		CMD="$$CMD --hyperparam_json $(HYPER)"; \
	fi; \
	CMD="$$CMD --num_trials $(NUM_TRIALS) $(RUN_ARGS)"; \
	echo "Running: $(PYTHON) code/run_piecewise.py $$CMD"; \
	$(PYTHON) code/run_piecewise.py $$CMD

run_linear:
	@CMD=""; \
	if [ -n "$(HYPER)" ]; then \
		CMD="$$CMD --hyperparam_json $(HYPER)"; \
	fi; \
	CMD="$$CMD --num_trials $(NUM_TRIALS) $(RUN_ARGS)"; \
	echo "Running: $(PYTHON) code/run_linear.py $$CMD"; \
	$(PYTHON) code/run_linear.py $$CMD

tune_run_piecewise:
	$(PYTHON) -m code.tune_and_run piecewise --num_trials $(NUM_TRIALS) $(TUNE_ARGS) $(RUN_ARGS)

tune_run_linear:
	$(PYTHON) -m code.tune_and_run linear --num_trials $(NUM_TRIALS) $(TUNE_ARGS) $(RUN_ARGS)
