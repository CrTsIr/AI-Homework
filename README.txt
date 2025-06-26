大作业的主题是基于 AI 实现一些帮助算法竞赛训练的功能。

下面按逻辑顺序介绍各个文件的功能。

requirements.txt 内是 pip 的依赖。

conda_list.txt 内是 conda 的依赖。

.env 储存了 api_key 与 url，可以直接修改。

以 knowledge 为前缀的 py 文件负责爬取网络信息及生成知识库：

knowledge_crawler.py 用于爬取网站 https://yhx-12243.github.io/OI-transit/，knowledge_arrange.py 用于整理，若爬取网站不同不大可复用。

knowledge_arrange.py 用初等算法整理爬虫获得的信息，比较枯燥，可以略过，最终的成果是所有以 solution 为后缀的 txt。

knowledge_generator.py 从 test_solution.txt 生成知识库并储存于 knowledge_export.json，若确有需要。

teach.py 需要输入题意文件和题解文件，然后可以开启对话。AI 会基于用户想题的进度给予一些提示，并会尝试通过知识库来展现出一些题目之间的共性。

draw.py 文件需要输入题意文件，然后可以开始对话。AI 集成了画树，图，序列的三种工具，并会基于题意以及上下文画一些和题目有关的图辅助思考。
