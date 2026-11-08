# Стратег Фабрики продуктов — единая точка входа для команд.
# Запуск: `make help`

SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

UV ?= uv
PYTHON ?= python3

.PHONY: help setup doctor run demo test test-slow eval lint format format-check \
        typecheck ci sources log cost clean wiki-index bench mcp web materials knowledge \
        lexicon lexicon-oov contracts-check scenarios-check demo-doc replay \
        semantic-bench \
        translate \
        canon tasks-check docs-check next-task

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
	    || echo "нет моделей"
	@$(UV) run python -c "from pf.config import settings; import sys; \
	  names = []; \
	  import httpx; \
	  r = httpx.get(settings.ollama_base_url.rstrip('/') + '/api/tags', timeout=5); \
	  names = [x['name'] for x in r.json().get('models', [])]; \
	  want = settings.ollama_model; \
	  ok = any(n == want or n.startswith(want.split(':')[0]) for n in names); \
	  print('Модель по умолчанию: ' + want + (' — установлена' if ok else ' — НЕ УСТАНОВЛЕНА, выполните: ollama pull ' + want)); \
	  sys.exit(0 if ok else 1)" \
	  || echo "  → выполните: ollama pull $(shell $(UV) run python -c 'from pf.config import settings; print(settings.ollama_model)')"
	@echo -n "Корпус:      "; \
	  if [ -d data/wiki ] && [ -n "$$(ls data/wiki/*.json 2>/dev/null)" ]; then \
	    echo "$$(ls data/wiki/*.json | wc -l) разделов"; \
	  else echo "НЕТ — проверьте data/wiki/"; exit 1; fi
	@echo -n "Промпты:     "; \
	  echo "$$(ls prompts/roles/*.md 2>/dev/null | wc -l) ролей"
	@echo -n "Материалы:   "; \
	  $(UV) run python -c "from pf.materials import load_materials; \
	  m = load_materials(); \
	  print(f'{len(m)} файлов' if m else 'нет (каталог project/ пуст)')"
	@echo -n "MCP:         "; \
	  if [ -n "$$(PF_MCP_DIR=$${PF_MCP_DIR:-../mcp-servers} ls ../mcp-servers 2>/dev/null)" ]; then \
	    echo "серверы найдены — список: pf mcp"; \
	  else echo "каталог серверов не найден (PF_MCP_DIR) — базовая работа не требует MCP"; fi
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
	@$(UV) run ruff check src/ tests/

eval: ## Выполнить eval-сценарии (правила поведения агента)
	@$(UV) run pf eval

format: ## Отформатировать код (ruff format)
	@$(UV) run ruff format src/ tests/

format-check: ## Проверить форматирование, не меняя файлы
	@$(UV) run ruff format --check src/ tests/

typecheck: ## Проверить типы (mypy --strict)
	@$(UV) run mypy src/

contracts-check: ## Проверить контракты: реестры против кода
	@$(UV) run python scripts/contracts_check.py

scenarios-check: ## Проверить, что сценарии покрыты тестами
	@$(UV) run pytest tests/ -m "not slow" -q -k "scenario"

demo-doc: ## Пересобрать demo/RESULT.md из журнала: make demo-doc SESSION=demo-v4
	@test -n "$(SESSION)" || (echo "Укажите сессию: make demo-doc SESSION=..." && exit 1)
	@$(UV) run python scripts/build_demo_doc.py --session "$(SESSION)"

ci: ## Полный прогон проверок
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory format-check
	@$(MAKE) --no-print-directory typecheck
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory eval
	@$(MAKE) --no-print-directory contracts-check
	@$(MAKE) --no-print-directory canon
	@$(MAKE) --no-print-directory tasks-check
	@$(MAKE) --no-print-directory docs-check

canon: ## Сверить числа канона (docs/canon.py) с деревом репозитория
	@$(UV) run python docs/canon.py

docs-check: ## Проверить поставляемые документы: числа, make-цели, пути
	@$(UV) run python scripts/docs_check.py

tasks-check: ## Проверить индекс задач и карточки: статусы, критерии, доказательства
	@$(UV) run python scripts/tasks_gate.py

next-task: ## Выбрать 1–2 задачи на сессию по RICE
	@$(UV) run python scripts/next_task.py

materials: ## Показать материалы проекта (документы, таблицы, данные)
	@$(UV) run pf materials

knowledge: ## Показать хранилище знаний (выгрузки, источники, хеши)
	@$(UV) run pf knowledge

lexicon: ## Показать словарь близости и очередь распознания слов
	@$(UV) run pf lexicon stats

lexicon-oov: ## Показать незнакомые слова, ждущие распознания
	@$(UV) run pf lexicon oov

fetch-data: ## Собрать открытые данные в хранилище знаний
	@$(UV) run python scripts/fetch_open_data.py

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

replay: ## A/B проверки оснований по сохранённым журналам (make replay SESSIONS="demo-v7 demo-v9")
	@$(UV) run python scripts/replay_verification.py $(if $(SESSIONS),--sessions $(SESSIONS),)

semantic-bench: ## Замер семантики: TF-IDF без моделей (make semantic-bench METHODS="tfidf rubert sbert")
	@$(UV) run python scripts/semantic_bench.py $(foreach m,$(or $(METHODS),tfidf),--method $(m))

mcp: ## Инструменты MCP-сервера (make mcp SERVER=static-parser)
	@$(UV) run pf mcp $(or $(SERVER),static-parser)

translate: ## Перевести английский материал (make translate FILE=путь или TEXT="...")
	@test -n "$(FILE)$(TEXT)" || (echo "Укажите источник: make translate FILE=путь | TEXT=\"...\"" && exit 1)
	@$(UV) run pf translate $(if $(FILE),$(FILE),--text "$(TEXT)")

web: ## Поиск в интернете и разбор страниц (make web QUERY="отток абонентов")
	@test -n "$(QUERY)" || (echo "Укажите запрос: make web QUERY=\"...\"" && exit 1)
	@$(UV) run pf web "$(QUERY)" --parse --limit 3

clean: ## Удалить кэши и артефакты прогона
	@rm -rf .pytest_cache .ruff_cache .mypy_cache
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "Готово (состояние сессий и журналы сохранены)"
