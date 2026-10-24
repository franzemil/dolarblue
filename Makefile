# Makefile — pipeline del informe final (dólar blue Bolivia)
#
# Uso desde 01_intro/:
#     make -C final_project render     # lo de todos los días
#     make -C final_project all        # + refresco de datos (cuota API)
#
# Uso desde final_project/:
#     make render

REPORT := report

.PHONY: help render data process diagrams figures pdf spell nbserver all clean dashboard dashboard-assets

help: ## Muestra los targets disponibles
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

render: diagrams figures pdf ## Diagramas + figuras + PDF (comando diario)

data: ## Refresca datos: main.py update-dataset (manifiesto 24h; extra args con ARGS="--force")
	uv run python main.py update-dataset $(ARGS)

process: ## Procesa snapshots en disco (replay JSON + LLM Gaceta); extra args con ARGS
	uv run python main.py process $(ARGS)

figures: ## Regenera figuras/tablas desde los CSV vigentes (sin llamar la API)
	uv run python $(REPORT)/make_figures.py

diagrams: ## Compila report/diagrams/*.mmd -> figures/diagram_*.pdf (mmdc)
	@mkdir -p $(REPORT)/figures
	@for d in $(REPORT)/diagrams/*.mmd; do \
		[ -e "$$d" ] || continue; \
		name=$$(basename "$$d" .mmd); \
		echo "  diagrams/$$name.mmd -> figures/diagram_$$name.pdf"; \
		mmdc -i "$$d" -o "$(REPORT)/figures/diagram_$$name.pdf" || exit 1; \
	done

pdf: ## Compila el informe con Tectonic -> report/build/main.pdf
	@mkdir -p $(REPORT)/build
	cd $(REPORT) && tectonic -o build main.tex

spell: ## Advertencias de ortografía (aspell es, modo TeX) — nunca bloquea
	@for f in $(REPORT)/main.tex $(REPORT)/sections/*.tex; do \
		bad=$$(aspell --lang=es --mode=tex \
			--add-tex-command='label p' \
			--add-tex-command='input p' \
			--add-tex-command='texttt p' \
			--add-tex-command='href po' \
			--add-tex-command='ref p' \
			--add-tex-command='hyperref po' \
			--add-tex-command='includegraphics op' \
			--add-tex-command='definecolor ppp' \
			--add-tex-command='sisetup p' \
			--personal=$(CURDIR)/$(REPORT)/dict/es_personal.dic --list < "$$f" | sort -u); \
		if [ -n "$$bad" ]; then \
			echo "\033[33m[$$f] palabras no reconocidas:\033[0m"; \
			echo "$$bad" | sed 's/^/    /'; \
		fi; \
	done; \
	echo "spell: revisa las palabras arriba (si hay) y agrega jerga a report/dict/es_personal.dic"

dashboard: ## Lanza el dashboard Streamlit (http://localhost:8501)
	uv run streamlit run dashboard/app.py

dashboard-assets: ## Regenera los parquet que consume el dashboard
	uv run python dashboard/scripts/export_assets.py

nbserver: ## Lanza jupyter lab con raíz en modelado/ (conectar VS Code: "Existing Jupyter Servers")
	uv run jupyter lab --notebook-dir=modelado --no-browser

all: data render ## Refresco completo: datos + diagramas + figuras + PDF

clean: ## Elimina salidas de compilación (report/build)
	rm -rf $(REPORT)/build
