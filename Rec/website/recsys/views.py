from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

types = ['未知', '动作', '冒险', '动漫', '儿童', '喜剧', '犯罪',
         '纪录片', '戏剧', '奇幻', '黑暗', '恐怖', '音乐', '神秘',
         '浪漫', '科幻', '惊悚', '战争', '西部']
type_e = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
uid = settings.UID

def hello_world(request):
    print(request.GET)
    return HttpResponse("Hello World")

def get_data():
    # 读文件
    u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    users = pd.read_csv(settings.BASE_DIR + '/ml-100k/u.user', sep='|', names=u_cols,
                        encoding='latin-1', parse_dates=True)
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv(settings.BASE_DIR + '/ml-100k/u.data', sep='\t', names=r_cols,
                          encoding='latin-1')
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url',
              'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
              'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
              'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movies = pd.read_csv(settings.BASE_DIR + '/ml-100k/u.item', sep='|', names=m_cols, encoding='latin-1')
    no_posters = np.loadtxt(settings.BASE_DIR + '/ml-100k/no_use.txt')
    movies = movies[~movies['movie_id'].isin(no_posters)]  # 剔除无用电影
    # 连接表
    movie_ratings = pd.merge(movies, ratings)
    df = pd.merge(movie_ratings, users)

    return users, ratings, movies, movie_ratings, df

def user_based(users, ratings, movie_ratings):# 获取user-based推荐
        # 评分矩阵转置
    watch_matrix = ratings.pivot_table(index=['user_id'], columns=['movie_id'], values='rating').reset_index(drop=True)
    watch_matrix.fillna(0, inplace=True)
        # 用户余弦相似度
    movie_similarity = 1 - pairwise_distances(watch_matrix, metric="cosine")
    np.fill_diagonal(movie_similarity, 0)  # Filling diagonals with 0s for future use when sorting is done
    watch_matrix = pd.DataFrame(movie_similarity)
        # 获取相似用户
    user_id = uid
    users['similarity'] = watch_matrix.iloc[user_id-1]
        # 评分个数大于200的地电影id
    movie_stats = movie_ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})
    min_50 = movie_stats['rating']['size'] >= 200
    min_50 = movie_stats[min_50].index
        # 生成推荐
    u_watched = movie_ratings[movie_ratings['user_id'] == user_id]
    u_watched_id = u_watched['movie_id']  # 该用户看过的电影id
    sim_u = users.sort_values(["similarity"], ascending=False)[1:2]['user_id']  # 相似用户id
    rec = []
    for su in sim_u:
        sim_u_watched = movie_ratings[movie_ratings['user_id'] == su]
        for suw in sim_u_watched.values:
            if suw[0] not in u_watched_id:
                if suw[0] in min_50:
                    rec.append(suw)
    return rec

def item_based(ratings, movies, m, n):
        # 评分矩阵
    ratings_matrix = ratings.pivot_table(index=['movie_id'], columns=['user_id'], values='rating').reset_index(
        drop=True)
    ratings_matrix.fillna(0, inplace=True)
        # 电影余弦相似度矩阵
    movie_similarity = 1 - pairwise_distances(ratings_matrix, metric="cosine")
    np.fill_diagonal(movie_similarity, 0)  # Filling diagonals with 0s for future use when sorting is done
    ratings_matrix = pd.DataFrame(movie_similarity)
        # 获取推荐
    user_inp = m
    inp = movies[movies['movie_id'] == user_inp].index.tolist()
    inp = inp[0]
    movies['similarity'] = ratings_matrix.iloc[inp]
    sim_movie = movies.sort_values( ["similarity"], ascending = False )[1:n+1]
    r = []
    for s in sim_movie.values:
        t = []
        t.append('/recsys/dilates?m=' + str(s[0]))
        t.append('/static/posters/'+str(s[0])+'.png')
        t.append(s[0]) # movie_id
        t.append(s[1][:-6]) # title
        t.append(str(s[24]*100)[:4]+'%') # similarity
        r.append(t)
    return r

def type_based(pred, n):
    _, _, movies, _, _ = get_data()
    rec = movies[movies[type_e[pred-1]]==1].sample(n=n)
    r = []
    for item in rec.values:
        ri = {'poster': '/static/posters/' + str(item[0]) + '.png', 'title': item[1][:-6]}
        r.append(ri)
    return r

def get_hot(df, n):# 获取人气电影
        # 获取高评分电影且评分个数大于50
    movie_stats = df.groupby(['title', 'movie_id']).agg({'rating': [np.size, np.mean]})
    min_50 = movie_stats['rating']['size'] >= 50
    hot_movies = movie_stats[min_50].sort_values([('rating', 'mean')], ascending=False).head(n)
    hot = hot_movies.index  # 获取title和id
    h = []
    for i in range(n):
        h.append([hot[i][0][:-6], '/recsys/dilates?m='+str(hot[i][1])])  # 除去title中的年份
    return h

def get_new(movies, n):# 获取最新电影
    new = movies.sort_values('release_date', ascending=True).head(n)
    n = []
    for nn in new.values:
        type_i = []
        for i in range(1, 20):
            if nn[i + 4] == 1:
                type_i.append(types[i - 1])
        t = {'poster': '/static/posters/' + str(nn[0]) + '.png', 'rank': i,
             'title': nn[1][:-6], 'type': type_i, 'url': '/recsys/dilates?m='+str(nn[0])}
        n.append(t)
    return n

def index(request):
    users, ratings, movies, movie_ratings, df = get_data()

    # 获取人气电影
    h = get_hot(df, 20)

    # 获取最新电影
    n = get_new(movies, 8)

    # 获取user-based推荐
    rec = user_based(users, ratings, movie_ratings)
        # 装入结果
    r = []
    for ii in range(10):
        type_i = []
        for i in range(1, 20):
            if rec[ii][i + 4] == 1:
                type_i.append(types[i - 1])
        t = {'poster': '/static/posters/' + str(rec[ii][0]) + '.png',
             'title': rec[ii][1][:-6], 'type': type_i, 'url': '/recsys/dilates?m='+str(rec[ii][0])}
        r.append(t)

    result = {'hot': h, 'new': n, 'rec': r}
    return render(request, 'index.html', result)

def poster_rec(request):
    m = request.GET.get('m')
    if m:
        m = int(m)
    else:
        m = 2

    # 获取原图分类结果
    cols = ['movie_id', 'pred_type', 'correct', 'title']
    preds = pd.read_csv(settings.BASE_DIR + '/ml-100k/pred.txt', sep='\t', names=cols,
                        encoding='latin-1')
    pred = preds[preds['movie_id'] == m]['pred_type'].values[0]
    correct = preds[preds['movie_id'] == m]['correct'].values[0]
    if correct == 1:
        correct = '正确'
    else:
        correct = '错误'

    # 获取攻击样本分类结果
    cols = ['movie_id', 'ae_pred_type']
    ae_preds = pd.read_csv(settings.BASE_DIR + '/ml-100k/FGSM.txt', sep='\t', names=cols,
                        encoding='latin-1')
    ae_pred = ae_preds[ae_preds['movie_id'] == m]['ae_pred_type'].values[0]

    # 获取原图推荐
    rec = type_based(pred, 6)

    # 获取攻击样本推荐
    ae_rec = type_based(ae_pred, 6)

    result = {'poster': '/static/posters/'+str(m)+'.png', 'pred_type': types[pred-1],
              'correct': correct, 'ae_poster': '/static/FGSM/'+str(m)+'.png',
              'ae_pred_type': types[ae_pred-1], 'rec': rec, 'ae_rec': ae_rec}
    return render(request, 'poster_rec.html', result)

def dilates(request):
    result = {}
    m = request.GET.get('m')
    if m:
        m = int(m)
        _, ratings, movies, _, df = get_data()
        # 获取人气电影
        h = get_hot(df, 10)
        # 获取最新电影
        n = get_new(movies, 10)
        # 获取该电影title
        title = str(movies[movies['movie_id']==m]['title'].values)[2:-2]
        release_date = str(movies[movies['movie_id']==m]['release_date'].values)[2:-2]
        url = str(movies[movies['movie_id']==1]['imdb_url'].values)[2:-2]
        # 获取item-based推荐
        r = item_based(ratings, movies, m, 6)

        result = {'poster': '/static/posters/'+str(m)+'.png', 'title': title,
            'release_date': release_date, 'url': url, 'rec': r, 'new': n,
            'hot': h, 'poster_rec': '/recsys/poster_rec?m='+str(m)}
    return render(request, 'dilates.html', result)

def movie(request):
    tp = request.GET.get('t')
    if tp:
        t = int(tp)-1 # 获取电影种类
        _, _, movies, _, _ = get_data()
        movie_t = movies.loc[movies[type_e[t]] == 1].head(25) # 找到该种类电影
        t = []
        for nn in movie_t.values: # 生成数据
            type_i = []
            for i in range(1, 20):
                if nn[i + 4] == 1:
                    type_i.append(types[i - 1])
            item = {'poster': '/static/posters/' + str(nn[0]) + '.png',
                    'title': nn[1][:-6], 'type': type_i, 'url': '/recsys/dilates?m='+str(nn[0])}
            t.append(item)
        result = {'t': t, 'cur_t': types[int(tp)-1]}
        return render(request, 'movie.html', result)
    else:
        _, _, movies, _, _ = get_data()
        t = []
        for nn in movies.head(25).values:  # 生成数据
            type_i = []
            for i in range(1, 20):
                if nn[i + 4] == 1:
                    type_i.append(types[i - 1])
            item = {'poster': '/static/posters/' + str(nn[0]) + '.png',
                    'title': nn[1][:-6], 'type': type_i, 'url': '/recsys/dilates?m='+str(nn[0])}
            t.append(item)
        result = {'t': t, 'cur_t': '全部'}
        return render(request, 'movie.html', result)
