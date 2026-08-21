.PHONY: check build

check: ## run all checks (markdownlint + check-posts)
	pre-commit run --all-files

build: ## local build verification
	hugo --ignoreCache --cleanDestinationDir
