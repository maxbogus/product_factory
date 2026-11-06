# Стратег Фабрики продуктов — единая точка входа для команд.
# Запуск: `make help`

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python3

.PHONY: help setup doctor run demo test test-slow eval lint format format-check \
        typecheck ci sources log cost clean wiki-index bench mcp web materials

help: ## Показать список доступных команд
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Демонстрация:  make demo"

setup: ## Установить зависимости (uv sync --extra dev)
	@echo "==> Установка зависимостей"
	@$(UV) sync --extra dev
	@echo ""
	@echo "Готово. Дальше: make doctor"

doctor: ## Проверить, что всё готово к работе
	@echo "==> Проверка окружения"
	@echo -n "Python:      "; $(PYTHON) --version
	@echo -n "Ollama:      "; \
	  if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then echo "доступен"; \
	  else echo "НЕДОСТУПЕН — запустите 'ollama serve'"; exit 1; fi
	@echo -n "Модель:      "; \
	  curl -sf http://localhost:11434/api/tags 2>/dev/null \
	    | $(PYTHON) -c "import json,sys; m=[x['name'] for x in json.load(sys.stdin)['models']]; print(', '.join(m))" \
	    || echo "нет моделей — выполните 'ollama pull qwen2.5:7b'"
	@echo -n "Корпус:      "; \
	  if [ -d data/wiki ] && [ -n "$$(ls data/wiki/*.json 2>/dev/null)" ]; then \
	    echo "$$(ls data/wiki/*.json | wc -l) разделов"; \
	  else echo "НЕТ — проверьте data/wiki/"; exit 1; fi
	@echo -n "Промпты:     "; \
	  echo "$$(ls prompts/roles/*.md 2>/dev/null | wc -l) ролей"
	@echo ""
	@echo "Всё готово. Попробуйте: make demo"

run: ## Поставить задачу: make run QUESTION="вопрос"
	@test -n "$(QUESTION)" || (echo "Укажите вопрос: make run QUESTION=\"...\"" && exit 1)
	@$(UV) run pf run "$(QUESTION)"

demo: ## Прогнать демонстрационную задачу целиком
	@$(UV) run pf run "Почему падает удержание у мобильного тарифа и что с этим делать?"

test: ## Детерминированные тесты (без обращения к модели)
	@$(UV) run pytest tests/ -m "not slow" -q

test-slow: ## Тесты с живой моделью
	@$(UV) run pytest tests/ -m slow -q

lint: ## Проверить стиль (ruff)

eval: ## Выполнить eval-сценарии (правила поведения агента)
	@$(UV) run pf eval
	@$(UV) run ruff check src/ tests/

format: ## Отформатировать код (ruff format)
	@$(UV) run ruff format src/ tests/

format-check: ## Проверить форматирование, не меняя файлы
	@$(UV) run ruff format --check src/ tests/

typecheck: ## Проверить типы (mypy --strict)
	@$(UV) run mypy src/

ci: ## Полный прогон проверок
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory format-check
	@$(MAKE) --no-print-directory typecheck
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory eval

materials: ## Показать материалы проекта (документы, таблицы, данные)
	@$(UV) run pf materials

sources: ## Показать источники: make sources QUERY="запрос"
	@test -n "$(QUERY)" || (echo "Укажите запрос: make sources QUERY=\"...\"" && exit 1)
	@$(UV) run pf sources "$(QUERY)"

log: ## Показать журнал шагов
	@$(UV) run pf log

cost: ## Показать расход
	@$(UV) run pf cost

wiki-index: ## Показать статистику корпуса
	@$(PYTHON) -c "import sys; sys.path.insert(0,'src'); from pf.corpus import get_corpus; c=get_corpus(); print(f'статей: {len(c)}'); from collections import Counter; [print(f'  {s}: {n}') for s, n in Counter(a.section for a in c.articles).most_common()]"

bench: ## Замеры: агент против обычного чата (make bench MODELS="qwen3:8b")
	@$(UV) run python scripts/bench.py $(if $(MODELS),--models $(MODELS),) $(if $(REPEATS),--repeats $(REPEATS),)

mcp: ## Инструменты MCP-сервера (make mcp SERVER=static-parser)
	@$(UV) run pf mcp $(or $(SERVER),static-parser)

web: ## Поиск в интернете и разбор страниц (make web QUERY="отток абонентов")
	@test -n "$(QUERY)" || (echo "Укажите запрос: make web QUERY=\"...\"" && exit 1)
	@$(UV) run pf web "$(QUERY)" --parse --limit 3

clean: ## Удалить кэши и артефакты прогона
	@rm -rf .pytest_cache .ruff_cache .mypy_cache
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "Готово (состояние сессий и журналы сохранены)"
