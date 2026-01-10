.PHONY: help install build publish clean test lint format dev release-patch release-minor release-major

# 获取当前版本
VERSION := $(shell grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# 默认目标
help:
	@echo "Bilibili-Captions Makefile"
	@echo ""
	@echo "当前版本: $(VERSION)"
	@echo ""
	@echo "可用命令:"
	@echo "  make install       - 本地安装包"
	@echo "  make build         - 构建发布包"
	@echo "  make publish       - 发布到 PyPI"
	@echo "  make clean         - 清理构建文件"
	@echo "  make test          - 运行测试"
	@echo "  make lint          - 代码检查"
	@echo "  make format        - 代码格式化"
	@echo "  make dev           - 安装开发依赖"
	@echo ""
	@echo "版本管理:"
	@echo "  make release-patch  - 补丁版本 (x.x.X → x.x.Y)"
	@echo "  make release-minor  - 次版本 (x.Y.x → x.Z.0)"
	@echo "  make release-major  - 主版本 (X.x.x → Y.0.0)"

# 本地安装
install:
	uv sync

# 构建发布包
build:
	rm -rf dist/
	uv build

# 发布到 PyPI
publish: build
	UV_PUBLISH_TOKEN="${UV_PUBLISH_TOKEN}" uv publish

# 清理构建文件
clean:
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# 运行测试
test:
	uv run pytest tests/

# 代码检查
lint:
	uv run ruff check .

# 代码格式化
format:
	uv run ruff format .

# 安装开发依赖
dev:
	uv sync --dev

# 版本管理 - 补丁版本 (0.0.1 → 0.0.2)
release-patch:
	@echo "当前版本: $(VERSION)"
	@new_version=$$(echo $(VERSION) | awk -F. '{print $$1"."$$2"."$$3+1}'); \
	sed -i '' "s/version = \"$(VERSION)\"/version = \"$$new_version\"/" pyproject.toml; \
	git add pyproject.toml; \
	git commit -m "chore: bump version to $$new_version"; \
	git tag -a "v$$new_version" -m "Release $$new_version"; \
	git push origin main; \
	git push origin "v$$new_version"; \
	echo "已发布 $$new_version"

# 版本管理 - 次版本 (0.1.0 → 0.2.0)
release-minor:
	@echo "当前版本: $(VERSION)"
	@new_version=$$(echo $(VERSION) | awk -F. '{print $$1"."$$2+1".0"}'); \
	sed -i '' "s/version = \"$(VERSION)\"/version = \"$$new_version\"/" pyproject.toml; \
	git add pyproject.toml; \
	git commit -m "chore: bump version to $$new_version"; \
	git tag -a "v$$new_version" -m "Release $$new_version"; \
	git push origin main; \
	git push origin "v$$new_version"; \
	echo "已发布 $$new_version"

# 版本管理 - 主版本 (0.1.0 → 1.0.0)
release-major:
	@echo "当前版本: $(VERSION)"
	@new_version=$$(echo $(VERSION) | awk -F. '{print $$1+1".0.0"}'); \
	sed -i '' "s/version = \"$(VERSION)\"/version = \"$$new_version\"/" pyproject.toml; \
	git add pyproject.toml; \
	git commit -m "chore: bump version to $$new_version"; \
	git tag -a "v$$new_version" -m "Release $$new_version"; \
	git push origin main; \
	git push origin "v$$new_version"; \
	echo "已发布 $$new_version"
