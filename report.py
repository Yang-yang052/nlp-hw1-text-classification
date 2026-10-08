"""Generate the expanded report entirely from measured experiment evidence."""
import json
import os
import re
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

METHODS = ['binary_bow','word_frequency','glove','word2vec_ag','word2vec_nyt','bert']
NAMES = {'binary_bow':'Binary BoW','word_frequency':'Word Frequency','glove':'GloVe',
         'word2vec_ag':'Word2Vec AG','word2vec_nyt':'Word2Vec NYT','bert':'BERT'}


def build():
    root = Path('results')
    required = METHODS + ['dataset','split','analysis','environment']
    missing = [n for n in required if not (root/(n+'.json')).exists()]
    if missing:
        raise SystemExit('Missing real experiment records: '+', '.join(missing))
    load = lambda n: json.loads((root/(n+'.json')).read_text(encoding='utf-8'))
    records = {n:load(n) for n in METHODS}
    data, split, analysis, env = [load(n) for n in ['dataset','split','analysis','environment']]
    font = Path(os.environ.get('REPORT_FONT','C:/Windows/Fonts/simhei.ttf'))
    if not font.exists():
        raise SystemExit('Set REPORT_FONT to a Chinese TrueType font file.')
    pdfmetrics.registerFont(TTFont('ReportChinese',str(font)))
    body = ParagraphStyle('body',fontName='ReportChinese',fontSize=10,leading=16,spaceAfter=7,wordWrap='CJK')
    title = ParagraphStyle('title',parent=body,fontSize=21,leading=28,spaceAfter=14)
    heading = ParagraphStyle('heading',parent=body,fontSize=14,leading=21,spaceBefore=9,spaceAfter=8,keepWithNext=True)
    sub = ParagraphStyle('sub',parent=body,fontSize=11,leading=16,spaceBefore=5,spaceAfter=5,keepWithNext=True)
    cell = ParagraphStyle('cell',parent=body,fontSize=9,leading=13,spaceAfter=0)
    quote = ParagraphStyle('quote',parent=body,fontSize=8.5,leading=13,textColor=colors.HexColor('#404040'),spaceAfter=5)
    code = ParagraphStyle('code',parent=body,fontSize=8.3,leading=12,spaceAfter=2)
    story = []
    def p(text,style=body):
        story.append(Paragraph(escape(str(text)),style))
    def table(rows,widths=None):
        t = Table([[Paragraph(escape(str(v)),cell) for v in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
            ('GRID',(0,0),(-1,-1),.45,colors.HexColor('#d9d9d9')),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),
            ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,9)])
    def chapter(text):
        story.append(PageBreak())
        p(text,heading)
    def m(n,key='macro_f1'):
        return records[n]['test'][key]
    labels = split['labels']
    best = max(METHODS,key=lambda n:m(n))

    p('文本分类实验报告',title)
    p('词袋模型  平均词向量  BERT 微调',sub)
    p(f"本实验在统一 NYT 新闻划分上比较六种文本分类方法。Binary BoW 的测试 Accuracy 为 {m('binary_bow','accuracy'):.4f}、Macro-F1 为 {m('binary_bow'):.4f}，在本次实验中排名第一。BERT 完整训练三轮并按验证集选择检查点，其测试 Macro-F1 为 {m('bert'):.4f}。结果表明，完整新闻中的关键词信息对本任务有效，模型复杂度增加没有自动带来更高测试分数。")
    p('一 作业目标与实现范围',heading)
    table([['任务','表示方式','分类器','统一要求'],
        ['Task 1','Binary BoW 与 Word Frequency','Logistic Regression','两组词袋实验'],
        ['Task 2','GloVe AG Word2Vec NYT Word2Vec','Logistic Regression','100 维 平均词向量'],
        ['Task 3','google-bert/bert-base-uncased','BERT 分类头','长度 64 训练 3 轮']], [52,207,110,136])
    p('Task 1 按更正后的要求实现两种表示，不额外增加第三种词袋方法。全部六组实验报告 Accuracy 和 Macro-F1，最终评价使用相同的 NYT Test Set。AG News 的文本仅用于 Word2Vec 训练，不参与 NYT 分类标签学习。')
    p('二 数据来源与类别分布',heading)
    p('实验使用提供的 nyt.csv 和 ag.csv，分别包含 11,519 条带标签新闻和 90,000 条无标签文本。两份文件均不存在缺失文本或空文本；NYT 标签为 business、politics 和 sports。')
    rows = [['类别','全体','占比','训练','验证','测试']]
    for name in labels:
        rows.append([name,data['counts'][name],f"{data['counts'][name]/data['n']:.2%}"]+
                    [part[name] for part in data['split_class_counts']])
    rows.append(['合计',data['n'],'100%',*data['sizes']])
    table(rows,[90,75,80,85,85,90])
    p('三 统一划分和数据边界',heading)
    p('设置随机种子 42，通过 NumPy 随机置换全部 NYT 行，前 80% 为训练集，随后 10% 为验证集，剩余 10% 为测试集。整数截断后样本数分别为 9,215、1,152、1,152。未额外分层，全部方法读取同一份 split.json 行号。')
    p('训练集用于构建词表、训练 NYT Word2Vec、拟合标准化参数以及训练分类器；验证集仅选择 LR 的 C 和 BERT 检查点；测试集仅作最终评价。split.json 保存标签映射及原始 NYT 文件 SHA256，核验脚本检查行号无交集且覆盖全部样本。')

    chapter('四 文本表示与分类方法')
    p('4.1 统一预处理与两种词袋表示',sub)
    p('前五组方法先将文本转为小写，再用正则表达式提取英文单词、词内撇号和数字。不删除停用词，不使用 TF-IDF，不限制最大词表。词表完全由 NYT 训练集建立，验证集和测试集中的未登录词不增加词表维度。')
    p(f"两种词袋共有 {records['binary_bow']['vocabulary_size']:,} 个训练词。设词 i 在文档 d 中出现 c_i(d) 次，则 Binary BoW 定义为 x_i(d)=1 当 c_i(d)>0，否则为 0；Word Frequency 定义为 x_i(d)=c_i(d)。")
    table([['示例文本','词表顺序','Binary BoW','Word Frequency'],
           ['apple apple banana','apple banana','[1, 1]','[2, 1]']], [180,110,105,110])
    p('Binary BoW 保留出现与否，减少重复词频对权重的放大；Word Frequency 保留重复强度，但也更受文档长度影响。特征使用稀疏矩阵存储，避免为每篇新闻创建大规模稠密向量。')
    p('4.2 三种 100 维词向量',sub)
    table([['方法','词向量来源','NYT 标签的作用'],
           ['GloVe','glove.6B.100d 公开预训练文件','仅训练 LR 分类器'],
           ['Word2Vec AG','ag.csv 全部 90,000 条文本','仅训练 LR 分类器'],
           ['Word2Vec NYT','NYT 训练集 9,215 条文本','仅训练 LR 分类器']], [100,280,125])
    p('GloVe 词向量固定，不在 NYT 上更新；文件经 SHA256 验证。两种 Word2Vec 使用相同参数：Skip-gram、vector_size=100、window=5、min_count=2、negative=5、epochs=5、workers=1、seed=42。单线程与固定词哈希函数减少训练随机差异。')
    p('对于 n 个已登录词，文档向量为 e(d)=(e(w1)+...+e(wn))/n。重复出现的词按出现次数参与平均，未登录词不计入分母；全为未登录词时使用 100 维零向量。平均表示固定维度，但丢失词序和局部组合信息。')
    p('4.3 Logistic Regression 与参数选择',sub)
    p('前五组实验均采用多类 Logistic Regression，solver=lbfgs、max_iter=2000。C 是正则化强度的倒数，候选值为 0.1、1、10。每个候选模型只在训练集拟合，选择验证 Macro-F1 最大的 C；出现同分时保留先遍历的较小 C。')
    p('三种平均词向量先使用 StandardScaler 标准化，均值和标准差只在训练集拟合。词袋不标准化。选择 C 后不合并验证集重训，因此所有最终分类器均遵循相同训练边界。')
    p('4.4 评价指标',sub)
    p('Accuracy=正确预测数/测试样本数。对类别 k，Precision_k=TP_k/(TP_k+FP_k)，Recall_k=TP_k/(TP_k+FN_k)，F1_k=2×Precision_k×Recall_k/(Precision_k+Recall_k)。Macro-F1=(F1_business+F1_politics+F1_sports)/3。')
    p(f"sports 在测试集中占 {analysis['majority_baseline']['accuracy']:.2%}。始终预测 sports 的 Accuracy 为 {analysis['majority_baseline']['accuracy']:.4f}，Macro-F1 却只有 {analysis['majority_baseline']['macro_f1']:.4f}。因此必须同时关注少数类别的分类质量；实现将全部三类纳入宏平均，零分母按 0 处理。")

    chapter('五 词袋与词向量实验结果')
    p('5.1 NYT 测试集结果',sub)
    table([['方法','Accuracy','Macro-F1','误分类篇数']]+[
        [NAMES[n],f"{m(n,'accuracy'):.4f}",f"{m(n):.4f}",analysis['error_counts'][n]]
        for n in METHODS[:5]], [175,100,110,120])
    p(f"Binary BoW 与 Word Frequency 均错分 12 篇，Accuracy 相同，但 Macro-F1 分别为 {m('binary_bow'):.4f} 和 {m('word_frequency'):.4f}。相同总错误数可以分布在不同类别，Macro-F1 可以揭示 Accuracy 掩盖的差异。")
    p('5.2 验证集选择记录',sub)
    table([['方法','C 0.1 验证 F1','C 1 验证 F1','C 10 验证 F1','选中 C']]+[
        [NAMES[n]]+[f"{v['macro_f1']:.4f}" for v in records[n]['validation_trials']]+[records[n]['best_C']]
        for n in METHODS[:5]], [135,95,95,100,80])
    p('Binary BoW 选择较强正则化 C=0.1，而 Word Frequency 选择 C=10。两种特征的数值尺度和词频分布不同，因此相同的 C 不必适合两种表示。该选择完全依据验证结果，而非测试结果。')
    p('5.3 已登录词覆盖率',sub)
    table([['词向量方法','训练覆盖率','验证覆盖率','测试覆盖率','测试零向量文档']]+[
        [NAMES[n]]+[f"{v['token_coverage']:.2%}" for v in records[n]['coverage']]+[
            records[n]['coverage'][2]['empty_documents']] for n in METHODS[2:5]], [120,90,90,90,115])
    p('覆盖率按词出现次数计算：有对应向量的 token 数除以全部 token 数。三种表示在测试集均无全未登录词文档，因而零向量回退没有直接影响本次测试结果。覆盖率较高不等于更高分类分数，因为向量质量及均值压缩同样影响表示。')
    p(f"NYT Word2Vec 相比 AG Word2Vec 的 Macro-F1 高 {m('word2vec_nyt')-m('word2vec_ag'):.4f}。领域匹配可能带来帮助，但两种语料的文档长度、词频和规模同时不同，不能单独把差值归因于领域。GloVe 的 Macro-F1 为 {m('glove'):.4f}，高于两种自行训练的词向量。")
    p('两种词袋在本次任务上优于平均词向量。稀疏词袋能够直接保留新闻类别相关的词汇线索；100 维均值向量把大量词的信息压缩成同一表示，可能削弱细粒度关键词区分能力。该解释符合表示机制，但本实验未用消融验证其因果作用。')

    chapter('六 BERT 微调与序列截断分析')
    p('6.1 模型与训练实现',sub)
    table([['参数','设置'],['预训练模型','google-bert/bert-base-uncased'],['最大长度','64 含 CLS 与 SEP 文本最多 62 个内容 token'],
        ['训练轮数与 batch','完整 3 epochs batch_size=16'],['优化器','AdamW lr=2e-5 weight_decay=0.01'],
        ['梯度与设备','梯度范数裁剪 1 CPU 16 线程 bfloat16 混合精度'],['选择依据','每轮验证 Macro-F1 最优检查点']], [140,365])
    p('使用 BERT 自带的 uncased WordPiece 分词器，截断长新闻并填充至 64，提供 attention_mask。分类头从随机初始化开始，BERT 编码器和分类头的全部参数共同微调。模型切换到 eval 后进行验证，评价时关闭梯度和训练 dropout。')
    p('6.2 三轮训练与验证记录',sub)
    history = records['bert']['history']
    table([['Epoch','训练 loss','验证 Accuracy','验证 Macro-F1']]+[
        [h['epoch'],f"{h['loss']:.4f}",f"{h['accuracy']:.4f}",f"{h['macro_f1']:.4f}"] for h in history], [70,145,145,145])
    chosen = max(history,key=lambda h:h['macro_f1'])['epoch']
    trend = '最后一轮验证分数低于最佳轮次，这与过拟合可能性一致，但不能仅凭三轮记录证明唯一原因。' if history[-1]['macro_f1'] < max(h['macro_f1'] for h in history) else '最后一轮也取得了最佳验证分数，完整记录见上表。'
    p(f'训练 loss 从第一轮的 {history[0]["loss"]:.4f} 变为第三轮的 {history[2]["loss"]:.4f}。验证 Macro-F1 最高值为 {max(h["macro_f1"] for h in history):.4f}，来自第 {chosen} 轮。因此完整完成三轮训练后，使用第 {chosen} 轮检查点，测试 Accuracy={m("bert","accuracy"):.4f}，Macro-F1={m("bert"):.4f}。'+trend)
    p('6.3 实测文本长度与截断比例',sub)
    length = analysis['bert_length']
    p(f"NYT 正则分词后的文档长度中位数为 {analysis['word_length']['median']:.0f} 个词。BERT WordPiece 长度含特殊 token，中位数为 {length['median']:.0f}、均值为 {length['mean']:.1f}、95 分位数为 {length['p95']:.0f}，远大于作业规定的 64。")
    table([['集合','文档数','超过 64 的文档数','被截断比例']]+[
        [label,analysis['bert_truncation'][k]['n'],analysis['bert_truncation'][k]['truncated'],
         f"{analysis['bert_truncation'][k]['fraction']:.2%}"]
        for k,label in [('train','训练'),('validation','验证'),('test','测试')]], [105,100,175,125])
    p('测试集 1,151 篇被截断，只有 1 篇无需截断。词袋与平均词向量使用全文，而 BERT 仅使用新闻开头，因此实验中的文本可见范围不同。BERT 较低分数与信息截断相容，但本实验遵守长度 64 的要求，没有额外训练长序列模型，不能据此给出截断的因果效应。')

    chapter('七 六组方法的总体比较')
    table([['方法','Accuracy','Macro-F1','相对 Binary BoW 的 F1 差']]+[
        [NAMES[n],f"{m(n,'accuracy'):.4f}",f"{m(n):.4f}",f"{m(n)-m('binary_bow'):+.4f}"]
        for n in METHODS], [155,95,100,155])
    p('7.1 每个类别的 F1',sub)
    table([['方法','business','politics','sports']]+[
        [NAMES[n]]+[f"{records[n]['classification_report'][label]['f1-score']:.4f}" for label in labels]
        for n in METHODS], [160,115,115,115])
    p('所有模型的 sports F1 都较高，主要差异出现在 business 与 politics。Binary BoW 的 politics 召回率为 1.0000，152 篇真实 politics 新闻全部识别正确，但有 8 篇其他新闻被预测为 politics，因此其精确率为 0.9500。高召回不能替代精确率分析。')
    p('7.2 BERT 与 Binary BoW 的逐样本比较',sub)
    paired = analysis['paired_bert_bow']
    table([['同一测试文档的预测关系','数量'],['两者均正确',paired['both_correct']],
           ['只有 Binary BoW 正确',paired['bow_only_correct']],['只有 BERT 正确',paired['bert_only_correct']],
           ['两者均错误',paired['both_wrong']]], [350,155])
    p(f"BERT 独自识别正确 {paired['bert_only_correct']} 篇，Binary BoW 独自识别正确 {paired['bow_only_correct']} 篇。Binary BoW 总计错 {analysis['error_counts']['binary_bow']} 篇，BERT 错 {analysis['error_counts']['bert']} 篇，净差为 {analysis['error_counts']['bert']-analysis['error_counts']['binary_bow']} 篇；两者的互补关系可以从表中直接观察。")
    p('两个模型的 sports 行完全相同：859 篇预测正确、2 篇判为 business、2 篇判为 politics。新增错误来自 business 多错 4 篇及 politics 多错 9 篇。宏平均对三类赋相同权重，因而这些少数类差异会明显影响 Macro-F1。')

    chapter('八 混淆矩阵与错误分布')
    p('以下均为测试集原始计数。每张矩阵的行是真实类别、列是预测类别，顺序为 business、politics、sports。对角线代表正确预测；各行总数分别为 137、152、863。')
    matrices = []
    for n in METHODS:
        rows = [['真实 / 预测','business','politics','sports']]+[
            [label,*records[n]['confusion_matrix'][i]] for i,label in enumerate(labels)]
        mat = Table([[Paragraph(str(v),cell) for v in row] for row in rows],colWidths=[70,52,52,52])
        mat.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.5,colors.HexColor('#d9d9d9')),
            ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8eef4')),
            ('BACKGROUND',(0,1),(0,-1),colors.HexColor('#f3f5f7')),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
        matrices.append([Paragraph(NAMES[n],sub),mat])
    grid = Table([[matrices[i],matrices[i+1]] for i in [0,2,4]],colWidths=[252,253])
    grid.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),16)]))
    story.append(grid)
    p('Binary BoW 最常见的混淆方向是 business→politics，共 6 篇。BERT 的 business→politics 为 7 篇、politics→business 为 5 篇，说明商业政策与政治议题之间存在语义交叉。')
    p('误分类不是只由少数类样本少造成的：sports 新闻也可能以经济或法律事件开头。下一节展示真实案例，帮助区分关键词混淆、类别边界交叉和短序列可见范围的问题。')
    p('矩阵能定位错误方向，但不能证明单个词直接导致了预测；本实验未执行注意力解释、特征贡献分解或因果消融。')

    chapter('九 误分类案例分析')
    p('案例从预测导出文件与原始 CSV 对齐取得，行号为 nyt.csv 的零基数据行号。先选取每种错误方向在测试顺序中的首个案例，再展示其中六例；摘录只清理不可见控制字符用于排版，训练输入保持原样。')
    notes = {
        5363:'新闻讨论学生债务和学费，但开头同时出现伊拉克、阿富汗及参战经历。金融主题和政治词汇交叉，Binary BoW 将 business 判为 politics。',
        2371:'新闻涉及足球赛事观众的住宿安排，包含旅游、酒店和城市服务内容。全文词袋仍误判为 business，说明保留全文不能消除跨主题混淆。',
        750:'新闻主题是商店与内衣制造商涉嫌价格串谋；sports bras 是商品名称。BERT 将 business 判为 sports，运动相关词汇与体育新闻主题发生交叉。',
        5018:'新闻围绕运动员的刑事审判，数据标签为 politics。BERT 预测 sports，反映球员身份与法律事件所属类别之间的主题边界交叉。',
        8198:'新闻开头是一起凶杀案件，背景为橄榄球赛后的球迷冲突，数据标签为 sports。BERT 预测 politics；开头本身就同时包含犯罪与体育内容，不能仅归因于后文截断。',
        10166:'新闻讨论动物收容所执行新法规，同时强调改造成本。数据标签为 politics，BERT 预测 business，表明法规与经营成本信息可以激活不同类别线索。'}
    by_row = {x['row_id']:x for x in analysis['examples']}
    chosen_examples = [by_row[row] for row in [5363,2371,750,5018,8198,10166] if row in by_row]
    chosen_examples += [x for x in analysis['examples'] if x['row_id'] not in {e['row_id'] for e in chosen_examples}]
    for ex in chosen_examples[:6]:
        row = ex['row_id']
        p(f"案例 {row}  {NAMES[ex['method']]}  {ex['true']} → {ex['predicted']}",sub)
        excerpt = re.sub(r'[\x00-\x1f\x7f-\x9f]',' ',ex['excerpt'])[:160].rsplit(' ',1)[0]
        p('原文开头 '+excerpt+' ...',quote)
        p(notes.get(row,f"数据标签为 {ex['true']}，模型预测为 {ex['predicted']}。摘录可用于检查文章开头的主题线索，但解释完整错误仍需结合原始全文。"))
    p('这些案例是预测结果的事后定性分析，不代表全部错分文档，也不等价于模型因果解释。完整十例和六组逐样本预测保存在 analysis.json 与各组 predictions.csv。',quote)

    chapter('十 复现说明与结论')
    p('10.1 代码入口与实际运行',sub)
    p('克隆仓库后进入项目目录，使用 Python 3.12 创建虚拟环境并安装 requirements.txt。数据 nyt.csv 和 ag.csv 放在根目录。下列命令分开运行六组实验；完整运行说明见 运行说明.md。')
    for line in ['python -m venv .venv','.venv/Scripts/python.exe -m pip install -r requirements.txt',
                 '.venv/Scripts/python.exe download_glove.py','.venv/Scripts/python.exe experiment.py --task bow',
                 '.venv/Scripts/python.exe experiment.py --task embeddings --glove assets/glove.6B.100d.txt',
                 '.venv/Scripts/python.exe experiment.py --task bert --threads 16 --bf16',
                 '.venv/Scripts/python.exe analyze_results.py','.venv/Scripts/python.exe validate_results.py',
                 '.venv/Scripts/python.exe report.py']:
        p(line,code)
    p('Linux 或 macOS 将 Python 路径改为 .venv/bin/python。bfloat16 需设备支持，不支持时去掉 --bf16；本次实测使用 CPU 16 线程和 bfloat16。报告嵌入中文 TrueType 字体；非 Windows 环境通过 REPORT_FONT 指定字体路径。',quote)
    p('10.2 实际依赖版本与保存证据',sub)
    table([['依赖','实测版本','依赖','实测版本']]+[
        [a,env[a],b,env[b]] for a,b in [('numpy','pandas'),('scikit-learn','scipy'),
        ('gensim','torch'),('transformers','reportlab')]], [140,100,165,100])
    p('experiment.py 实现全部六组训练；test_core.py 检查词频、均值向量和划分语义；validate_results.py 重新计算 Accuracy、Macro-F1 和混淆矩阵，并核验原始数据 SHA256。results 保存划分、验证记录、分类报告和逐样本预测；assets 和大模型检查点不纳入 Git 仓库。')
    p('10.3 结论与局限',sub)
    p(f'在该数据与统一划分下，{NAMES[best]} 的 Macro-F1 最高。Word Frequency 保留频次，但未超过二值词袋；平均词向量保留语义相似性但压缩了全文关键词；BERT 能建模上下文，但受到长度 64 和训练轮数 3 的规定约束。')
    p('原始 NYT 有 72 条重复文本，训练与验证、训练与测试、验证与测试分别重叠 12、15、1 个不同文本。为遵循随机划分要求保留原始数据，这可能使评价偏乐观。本实验只运行一个随机种子，未进行多次运行、统计显著性检验或长序列消融；对模型机制的解释应视为有证据支持的可能原因。')
    p('参考资料',sub)
    p('Pennington 等 GloVe 2014。https://nlp.stanford.edu/projects/glove/',quote)
    p('Devlin 等 BERT 2019。https://aclanthology.org/N19-1423/',quote)
    p('模型来源 https://huggingface.co/google-bert/bert-base-uncased',quote)
    def footer(c,doc):
        c.setFont('ReportChinese',9)
        c.drawCentredString(297,24,str(doc.page))
    SimpleDocTemplate('实验报告.pdf',pagesize=(595.28,841.89),leftMargin=45,rightMargin=45,
                      topMargin=37,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)


if __name__ == '__main__':
    build()
