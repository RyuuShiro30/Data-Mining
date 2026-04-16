import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

dataset = [
    ['roti', 'susu', 'telur'],
    ['roti', 'mentega', 'telur'],
    ['susu', 'mentega', 'roti', 'telur'],
    ['roti', 'susu'],
    ['roti', 'mentega', 'telur']
]

te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)
df
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.5)
rules
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

plt.scatter(rules['support'], rules['confidence'])
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Confidence')
plt.show()