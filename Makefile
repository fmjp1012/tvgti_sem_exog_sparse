PYTHON ?= /Users/fmjp/venv/default/bin/python

.PHONY: help config tune_piecewise tune_linear run_piecewise run_linear piecewise linear

help:
	@echo "=========================================="
	@echo "シミュレーション設定は code/config.py で変更してください"
	@echo "=========================================="
	@echo ""
	@echo "利用可能なターゲット:"
	@echo "  make config           # 現在の設定を表示"
	@echo ""
	@echo "  make piecewise        # Piecewise: チューニング → シミュレーション"
	@echo "  make linear           # Linear: チューニング → シミュレーション"
	@echo ""
	@echo "  make tune_piecewise   # Piecewise: チューニングのみ"
	@echo "  make tune_linear      # Linear: チューニングのみ"
	@echo ""
	@echo "  make run_piecewise    # Piecewise: シミュレーションのみ"
	@echo "  make run_linear       # Linear: シミュレーションのみ"
	@echo ""
	@echo "※ すべての設定は code/config.py で一元管理されています"
	@echo "※ コマンドライン引数による設定変更は非推奨です"

config:
	$(PYTHON) code/config.py

piecewise:
	$(PYTHON) -m code.tune_and_run piecewise

linear:
	$(PYTHON) -m code.tune_and_run linear

tune_piecewise:
	@echo "チューニングのみ実行するには config.py の skip_simulation を True に設定してください"
	$(PYTHON) -m code.tune_and_run piecewise

tune_linear:
	@echo "チューニングのみ実行するには config.py の skip_simulation を True に設定してください"
	$(PYTHON) -m code.tune_and_run linear

run_piecewise:
	$(PYTHON) -m code.run_piecewise

run_linear:
	$(PYTHON) -m code.run_linear
