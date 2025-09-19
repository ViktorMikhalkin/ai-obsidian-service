# Changelog

## 0.1.0 (2025-09-19)


### Features

* **api:** add /answer endpoint with optional Ollama support and extractive fallback ([137b676](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/137b676d63c832164d1cf6db32705dc2c1125450))
* **api:** add /index/stats and /index/rebuild endpoints ([88220f8](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/88220f850e963a1bc63d4d8d36246fa0fc917c11))
* **ci:** add version file for release-please ([3613baa](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/3613baa9d92c147640d4b6ecf20472a4cccd1aee))
* **docs:** add OpenAPI export script (scripts/export_openapi.py) ([657ebdf](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/657ebdf665ac0cd85d13b7f2c03cdccaded5f565))
* **parser:** add EPUB parser ([d9263b3](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/d9263b3fdac56865ac768801a8bc68851ec14e74))


### Bug Fixes

* **ci:** add core scope to pr-title-check.yml ([845fd37](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/845fd37e77950345c52adaa6f58a8851eb39903e))
* **ci:** add type support to mypy ([091de1f](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/091de1f6616faff2b8e704f48860dbb6beec08d0))
* **ci:** change clear mypy cache command ([a03f1c0](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/a03f1c0dbcd4c3da3a22d20aabea133c80e4d9c4))
* **ci:** change clear mypy cache command ([013482d](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/013482da24817c46d802633ac238ecfb477c3910))
* **ci:** ensure code-quality job uses conda ([d0ecb9f](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/d0ecb9fc916f7266debfa4e2e0181fd92634d934))
* **ci:** ensure types-PyYAML dependency is installed ([02a3c18](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/02a3c182ed5f0e941c8802bea202d3e04c65c83c))
* **core:** fix mypy errors ([03d0b07](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/03d0b07087d54bf32d27033d72226a15f448d95c))
* **docs:** add core scope to PR template ([cc33733](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/cc337337a0bff76dffb808ad3369d6677c27b53e))
* **docs:** return content to README.md ([ce0639b](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/ce0639bc7a4ce4f45b664f2ce182f9f00c676e75))
* **faiss:** use loaded FAISS index, config-driven index_dir; fix env-core.yml name; update README ([e74abd1](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/e74abd18a75b2092af5ecc9375396b5e536ff334))
* **index:** add __init__.py to clarify package boundaries for mypy ([5bee187](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/5bee187df2b329b8641bd144d312f59461893a63))
* **repo:** add types-PyYAML dependency ([58cd7de](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/58cd7deefe3d5dfd62b7130dbd968c0c228e9f78))
* **repo:** downgrade version ([ef1f43e](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/ef1f43e7d6a6e19ebe2dfbdb859c9758ae6b780b))
* **repo:** fix code formatting ([6dff916](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/6dff916de124485b6164acf9f6f457b16f3511fa))
* **repo:** fix mypy checks ([29a163e](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/29a163edd810ff5e986067630614260a8e5f544b))
* **repo:** fix setup for CPU case ([6497f9b](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/6497f9b6ad7a6ff928cd575bbd9a18835aa9c4fe))


### Documentation

* **ci:** add CI/CD usage guide with release-please flow ([2b9ce6a](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/2b9ce6aa98566a299f6d6c9df264ff40a3b366f1))
* **docs:** add bug, feature, and task templates ([3aab777](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/3aab777853af059e035f30000d771449e1df4012))
* **docs:** add Code of Conduct and Security Policy ([c342029](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/c342029eda8dfc258103fb5408106538f1834985))
* **docs:** add contributing guide ([755952e](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/755952e9e564852643b0b962b1716fe5f64a39c5))
* **docs:** add pull request template with checklist ([06d0bd8](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/06d0bd82c6b61caba84d1b617ccf7b7997aa784e))
* **docs:** refresh README with setup steps, Make targets, and troubleshooting guidance. Closes [#37](https://github.com/ViktorMikhalkin/ai-obsidian-service/issues/37) ([a75e4a9](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/a75e4a9131e8cafced521480822c6c2bad2187e0))
* **docs:** revamp contributing guide with setup, CI parity, husky and PR guidelines. Closes [#40](https://github.com/ViktorMikhalkin/ai-obsidian-service/issues/40) ([54dc8c7](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/54dc8c71ba07125a3a8bb7f0f9b3a4564995c8f5))
* **repo:** add basic README ([64dd074](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/64dd074746712f135672f31c8a6c5f3e6c16f9bd))

## [0.2.0](https://github.com/ViktorMikhalkin/ai-obsidian-service/compare/ai-obsidian-service-v0.1.0...ai-obsidian-service-v0.2.0) (2025-09-15)


### Features

* **api:** add /answer endpoint with optional Ollama support and extractive fallback ([137b676](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/137b676d63c832164d1cf6db32705dc2c1125450))
* **api:** add /index/stats and /index/rebuild endpoints ([88220f8](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/88220f850e963a1bc63d4d8d36246fa0fc917c11))
* **ci:** add version file for release-please ([3613baa](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/3613baa9d92c147640d4b6ecf20472a4cccd1aee))
* **docs:** add OpenAPI export script (scripts/export_openapi.py) ([657ebdf](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/657ebdf665ac0cd85d13b7f2c03cdccaded5f565))
* **parser:** add EPUB parser ([d9263b3](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/d9263b3fdac56865ac768801a8bc68851ec14e74))


### Bug Fixes

* **ci:** add type support to mypy ([091de1f](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/091de1f6616faff2b8e704f48860dbb6beec08d0))
* **ci:** change clear mypy cache command ([a03f1c0](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/a03f1c0dbcd4c3da3a22d20aabea133c80e4d9c4))
* **ci:** change clear mypy cache command ([013482d](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/013482da24817c46d802633ac238ecfb477c3910))
* **ci:** ensure code-quality job uses conda ([d0ecb9f](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/d0ecb9fc916f7266debfa4e2e0181fd92634d934))
* **ci:** ensure types-PyYAML dependency is installed ([02a3c18](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/02a3c182ed5f0e941c8802bea202d3e04c65c83c))
* **faiss:** use loaded FAISS index, config-driven index_dir; fix env-core.yml name; update README ([e74abd1](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/e74abd18a75b2092af5ecc9375396b5e536ff334))
* **index:** add __init__.py to clarify package boundaries for mypy ([5bee187](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/5bee187df2b329b8641bd144d312f59461893a63))
* **repo:** add types-PyYAML dependency ([58cd7de](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/58cd7deefe3d5dfd62b7130dbd968c0c228e9f78))
* **repo:** fix code formatting ([6dff916](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/6dff916de124485b6164acf9f6f457b16f3511fa))
* **repo:** fix mypy checks ([29a163e](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/29a163edd810ff5e986067630614260a8e5f544b))
* **repo:** fix setup for CPU case ([6497f9b](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/6497f9b6ad7a6ff928cd575bbd9a18835aa9c4fe))


### Documentation

* **ci:** add CI/CD usage guide with release-please flow ([2b9ce6a](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/2b9ce6aa98566a299f6d6c9df264ff40a3b366f1))
* **docs:** add bug, feature, and task templates ([3aab777](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/3aab777853af059e035f30000d771449e1df4012))
* **docs:** add Code of Conduct and Security Policy ([c342029](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/c342029eda8dfc258103fb5408106538f1834985))
* **docs:** add contributing guide ([755952e](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/755952e9e564852643b0b962b1716fe5f64a39c5))
* **docs:** add pull request template with checklist ([06d0bd8](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/06d0bd82c6b61caba84d1b617ccf7b7997aa784e))
* **docs:** refresh README with setup steps, Make targets, and troubleshooting guidance. Closes [#37](https://github.com/ViktorMikhalkin/ai-obsidian-service/issues/37) ([a75e4a9](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/a75e4a9131e8cafced521480822c6c2bad2187e0))
* **docs:** revamp contributing guide with setup, CI parity, husky and PR guidelines. Closes [#40](https://github.com/ViktorMikhalkin/ai-obsidian-service/issues/40) ([54dc8c7](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/54dc8c71ba07125a3a8bb7f0f9b3a4564995c8f5))
* **repo:** add basic README ([64dd074](https://github.com/ViktorMikhalkin/ai-obsidian-service/commit/64dd074746712f135672f31c8a6c5f3e6c16f9bd))

## Changelog

> Managed by Release Please.
