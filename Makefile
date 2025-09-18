.PHONY: setup index report verify package

CONFIG ?= CONFIG.sample.yml
RUN ?= dev
SECTION ?= transport

setup:
	pip install -e .[test]

index:
	tpa index --config $(CONFIG)

report:
	tpa report --section $(SECTION) --config $(CONFIG) --run $(RUN)

verify:
	tpa verify --run $(RUN)

package:
	tpa package --run $(RUN)
