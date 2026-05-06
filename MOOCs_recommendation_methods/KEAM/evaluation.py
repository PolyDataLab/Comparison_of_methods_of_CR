import pandas as pd
from collections import defaultdict
import numpy as np






def calculate_metrics(test_df, recommendations_df, top_n=10):
    # Initialize metrics
    precision = 0
    recall = 0
    ndcg = 0
    hr_count = 0
    num_users = len(test_df['user'].unique())
    nums = len(test_df['user'])

    # Convert test set to dictionary format
    test_dict = test_df.groupby('user')['item'].apply(set).to_dict()

    # Generate top-n recommendations list for each user
    top_n_recommendations = (
        recommendations_df.groupby('user').head(top_n)
        .groupby('user')['item'].apply(list).to_dict()
    )

    # Calculate metrics
    for user, relevant_items in test_dict.items():
        if user in top_n_recommendations:
            recommended_items = top_n_recommendations[user]
            hr = False
            dcg = 0.0
            idcg = 0.0
            for rank, item in enumerate(recommended_items, start=1):
                if item in relevant_items:
                    hr = True
                    precision += 1 / top_n
                    recall += 1 / len(relevant_items)
                    dcg += 1 / np.log2(rank + 2)  # rank+2 because log2(1) is 0 and ranks start at 0
                    # hr_count += 1
                # Calculate IDCG part
                if rank <= len(relevant_items):
                    idcg += 1 / np.log2(rank + 2)  # Same as DCG for the 'ideal' case
            # Calculate nDCG for the user
            if hr == True:
                hr_count += 1
            ndcg += dcg / idcg if idcg > 0 else 0
    
    # Calculate average of metrics
    precision /= num_users
    recall /= num_users
    ndcg /= num_users
    hr = hr_count / num_users

    return {"Precision": precision, "Recall": recall, "HR": hr, "NDCG": ndcg}



# 加载测试集和推荐列表
recommendations_df = pd.read_csv('edm&ipsj/predictions_EXT_n200', sep='\t', names=['user', 'item', 'interaction'])
test_df = pd.read_csv('edm&ipsj/mooc_data/Data/mooc.test.rating', sep='\t', names=['user', 'item', 'score'])

# print(test_df.head())

# 计算指标
metrics = calculate_metrics(test_df, recommendations_df, top_n=10)
print(metrics)


# predictions_100: {'Precision': 0.03667068757539678, 'Recall': 0.16953242610019414, 'HR': 0.36670687575392036, 'NDCG': 0.1977685699057025}

# predictions_n195: {'Precision': 0.06342371636859473, 'Recall': 0.2402481332559992, 'HR': 0.6342371636859495, 'NDCG': 0.34666163754470625}

# predictions_ex_n195: {'Precision': 0.06677505638012994, 'Recall': 0.2508145980345732, 'HR': 0.6677505638013321, 'NDCG': 0.3737993074889112}

# predictions_EXT_n200: {'Precision': 0.06903026170870606, 'Recall': 0.26693344525244384, 'HR': 0.3732627051974616, 'NDCG': 0.21467312589541143}