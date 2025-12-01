# API 申请指南

## DeepSeek
- 官网：[https://platform.deepseek.com/](https://platform.deepseek.com/)
- 进入控制台，创建 API Key，复制后在侧边栏「模型配置」中填写。
- 建议为不同环境创建独立 Key，便于权限管理。

## 智谱 GLM
- 官网：[https://open.bigmodel.cn/](https://open.bigmodel.cn/)
- 注册并完成实名认证后，可在「API Keys」页面创建 Key。
- 选择所需的模型规格，例如 `glm-4`、`glm-4-plus` 等。

## 阿里云通义 Qwen
- 官网：[https://dashscope.aliyun.com/](https://dashscope.aliyun.com/)
- 登录后在控制台的 **API-KEY 管理** 页面创建 DashScope API Key。
- 官方提供 `qwen-plus`、`qwen-max`、`qwen-long`、`qwen2-72b-instruct`、`qwen-coder-plus` 等多种子模型，可按需选择。
- 获取 Key 后在侧栏填写，即可与 DeepSeek / GLM 一样使用对话与检索流程。

## 本地模型
- 在 `model/chat_models/` 目录放置兼容 OpenAI 协议的本地 LLM。
- 配置本地服务地址（默认为 `http://localhost:23333/v1`）即可使用。

> 获取 Key 之后，可随时在侧边栏修改，程序会自动重新初始化模型。

