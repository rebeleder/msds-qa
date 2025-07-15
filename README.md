使用方法

在根目录创建assets文件夹，放入mdsd某一个化学品的pdf文件

在根目录创建.env文件写入

```
OPENAI_API_KEY=
```

打开vscode，在终端输入

```
conda create -n msdsqa python=3.13.5
conda activate msdsqa
pip install -r requirements.txt
```

在vscode调试先运行msds_pipe，再运行智能体主入口
