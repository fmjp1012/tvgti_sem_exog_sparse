# ホスト名でローカル(Mac)とSSH先(サーバー)を自動判別
# .env ファイルがあれば読み込む (サーバーごとの設定用)
-include .env

HOSTNAME := $(shell hostname)

# Mac (ローカル) - uname -s は Darwin を返す
ifeq ($(findstring darwin,$(shell uname -s | tr A-Z a-z)),darwin)
    PYTHON ?= /Users/fmjp/venv/default/bin/python
# Linux (SSH先サーバー)
else
    # pyenvのshimがあれば優先的に使用 (python3を優先)
    ifneq ($(wildcard $(HOME)/.pyenv/shims/python3),)
        PYTHON ?= $(HOME)/.pyenv/shims/python3
    else ifneq ($(wildcard $(HOME)/.pyenv/shims/python),)
        PYTHON ?= $(HOME)/.pyenv/shims/python
    else
        # PYTHON変数が未設定の場合、利用可能なPythonを探す (python3 優先)
        PYTHON ?= $(shell which python3 2>/dev/null || which python 2>/dev/null)
    endif
endif
LOG_DIR ?= logs
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)

.PHONY: help config tune_piecewise tune_linear run_piecewise run_linear piecewise linear \
        bg_piecewise bg_linear bg_tune_piecewise bg_tune_linear bg_run_piecewise bg_run_linear \
        bg_status bg_stop

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
	@echo "=========================================="
	@echo "SSH切断後も継続するバックグラウンド実行:"
	@echo "=========================================="
	@echo "  make bg_piecewise     # Piecewise: バックグラウンドで実行"
	@echo "  make bg_linear        # Linear: バックグラウンドで実行"
	@echo "  make bg_run_piecewise # Piecewise: シミュレーションのみ (バックグラウンド)"
	@echo "  make bg_run_linear    # Linear: シミュレーションのみ (バックグラウンド)"
	@echo ""
	@echo "  make bg_status        # バックグラウンドジョブの状態確認"
	@echo "  make bg_stop          # バックグラウンドジョブを停止"
	@echo "  make bg_tail          # 最新のログをtail -f"
	@echo ""
	@echo "※ ログは $(LOG_DIR)/ に保存されます"
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

# ========================================
# SSH切断後も継続するバックグラウンド実行
# ========================================

$(LOG_DIR):
	mkdir -p $(LOG_DIR)

bg_piecewise: $(LOG_DIR)
	@echo "バックグラウンドで piecewise を開始します..."
	@echo "ログファイル: $(LOG_DIR)/piecewise_$(TIMESTAMP).log"
	@nohup $(PYTHON) -u -m code.tune_and_run piecewise > $(LOG_DIR)/piecewise_$(TIMESTAMP).log 2>&1 & echo $$! > $(LOG_DIR)/piecewise.pid
	@echo "PID: $$(cat $(LOG_DIR)/piecewise.pid)"
	@echo "ログ確認: tail -f $(LOG_DIR)/piecewise_$(TIMESTAMP).log"

bg_linear: $(LOG_DIR)
	@echo "バックグラウンドで linear を開始します..."
	@echo "ログファイル: $(LOG_DIR)/linear_$(TIMESTAMP).log"
	@nohup $(PYTHON) -u -m code.tune_and_run linear > $(LOG_DIR)/linear_$(TIMESTAMP).log 2>&1 & echo $$! > $(LOG_DIR)/linear.pid
	@echo "PID: $$(cat $(LOG_DIR)/linear.pid)"
	@echo "ログ確認: tail -f $(LOG_DIR)/linear_$(TIMESTAMP).log"

bg_run_piecewise: $(LOG_DIR)
	@echo "バックグラウンドで run_piecewise を開始します..."
	@echo "ログファイル: $(LOG_DIR)/run_piecewise_$(TIMESTAMP).log"
	@nohup $(PYTHON) -u -m code.run_piecewise > $(LOG_DIR)/run_piecewise_$(TIMESTAMP).log 2>&1 & echo $$! > $(LOG_DIR)/run_piecewise.pid
	@echo "PID: $$(cat $(LOG_DIR)/run_piecewise.pid)"
	@echo "ログ確認: tail -f $(LOG_DIR)/run_piecewise_$(TIMESTAMP).log"

bg_run_linear: $(LOG_DIR)
	@echo "バックグラウンドで run_linear を開始します..."
	@echo "ログファイル: $(LOG_DIR)/run_linear_$(TIMESTAMP).log"
	@nohup $(PYTHON) -u -m code.run_linear > $(LOG_DIR)/run_linear_$(TIMESTAMP).log 2>&1 & echo $$! > $(LOG_DIR)/run_linear.pid
	@echo "PID: $$(cat $(LOG_DIR)/run_linear.pid)"
	@echo "ログ確認: tail -f $(LOG_DIR)/run_linear_$(TIMESTAMP).log"

bg_status:
	@echo "=== バックグラウンドジョブの状態 ==="
	@ps aux | grep -E "python.*code\.(tune_and_run|run_)" | grep -v grep || echo "実行中のジョブはありません"
	@echo ""
	@echo "=== PIDファイル ==="
	@for f in $(LOG_DIR)/*.pid; do \
		if [ -f "$$f" ]; then \
			pid=$$(cat "$$f"); \
			if ps -p "$$pid" > /dev/null 2>&1; then \
				echo "$$f: PID $$pid (実行中)"; \
			else \
				echo "$$f: PID $$pid (終了済み)"; \
			fi; \
		fi; \
	done 2>/dev/null || echo "PIDファイルはありません"

bg_stop:
	@echo "バックグラウンドジョブを停止します..."
	@for f in $(LOG_DIR)/*.pid; do \
		if [ -f "$$f" ]; then \
			pid=$$(cat "$$f"); \
			if ps -p "$$pid" > /dev/null 2>&1; then \
				echo "Stopping PID $$pid..."; \
				kill "$$pid"; \
			fi; \
			rm -f "$$f"; \
		fi; \
	done 2>/dev/null || echo "停止するジョブはありません"

bg_tail:
	@latest=$$(ls -t $(LOG_DIR)/*.log 2>/dev/null | head -1); \
	if [ -n "$$latest" ]; then \
		echo "=== $$latest ==="; \
		tail -f "$$latest"; \
	else \
		echo "ログファイルがありません"; \
	fi
