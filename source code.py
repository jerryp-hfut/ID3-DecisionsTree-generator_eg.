from sklearn.tree import DecisionTreeClassifier #决策树分类器
from sklearn.tree import export_text #用于生成用于可视化决策树的文本表示
import pandas as pd #用于创建数据框对象

# 创建输入的数据表格，假设包含以下列：是否存在其他选择，饿否，价格，餐馆类型，餐馆顾客数量，等待时间和目标变量，存储到字典data中
data = {
    'other_choices_exist': ['1', '1', '0', '1', '1','0','0','0','0','1','1','0'], #以下数据列表中只有0/1的即表示“是/否”
    'hungry': ['1', '1', '0', '1', '0','1','0','1','0','1','0','1'],
    'price': ['3', '1', '1', '1', '3','2','1','2','1','3','1','1'],
    'restaurant_type': ['3', '1', '2', '1', '3','4','2','1','2','4','1','2'], #1代表中餐，2代表快餐，3代表法式，4代表意大利式
    'amount_of_customers': ['2', '3', '2', '3', '3','2','1','2','3','3','1','3'], #1代表无人，2代表有人，3代表客满
    'time_of_waiting': ['1', '3', '1', '2', '4','1','1','1','4','2','1','3'], #1代表0-10min，2代表10-30min，3代表30-60min，4代表>60min
    'target': ['1', '0', '1', '1', '0','1','0','1','0','0','0','1']
}

df = pd.DataFrame(data) #创建数据框对象

# 输入特征和目标变量
X = df[['other_choices_exist', 'hungry', 'price', 'restaurant_type', 'amount_of_customers', 'time_of_waiting']]
y = df['target'].values

# 创建决策树分类器，进行拟合
clf = DecisionTreeClassifier()
clf.fit(X, y)

# 输出决策树
tree_rules = export_text(clf, feature_names=X.columns.tolist())
print(tree_rules)