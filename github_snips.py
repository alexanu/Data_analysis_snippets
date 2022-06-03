
# Moving existing folder to github
    '''
    1. "git init -b main"
    2. "git add ."
    3. "git commit -m "First Commit" "
    4. Create empty new repo on Github without Readme.md
    5. "git remote add origin https://github.com/alexanu/alpaca_gcp.git"
    6. "git push origin main"
    '''

# Pushing changes to GH
    '''
    1. manually select file to stage
         or
       "git add ITR\data\template.py"
         or (stage everything)
       "git add -A"
    2) git commit -m "Adding fields for bonds and ETFs to indicators_list.py"
    3) git push
    '''

# check url or remote: git config --get remote.origin.url

# update local content from Github: git pull origin main

# download files from github
    import urllib.request
    from github import Github

    # Do this before: "set GITHUB_TOKEN=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    gh = Github(os.environ['GITHUB_TOKEN'])
    repo = gh.get_repo("os-climate/ITR").get_branch(branch="develop-pint-steel-projections")

    if not os.path.isdir("data"):
        os.mkdir("data")

    for filename in ['data/20220415 ITR Tool Sample Data.xlsx',
                    'data/OECM_EI_and_production_benchmarks.xlsx',
                    'utils.py']:
        if not os.path.isfile(filename):
            contents = repo.get_contents(f"examples/{filename}") # Get a specific content file
            urllib.urlretrieve(contents.download_url, filename) # Download file form ContenFile object info


# Parsing Github
    # $ pip install PyGithub
    # PyGithub documentation: https://buildmedia.readthedocs.org/media/pdf/pygithub/stable/pygithub.pdf


    from github import Github
    import pandas as pd
    import os

    from configparser import ConfigParser
    config = ConfigParser()
    config.read(Path_To_TKNS)
    GHLogin=config['GitHub']['Login']
    GHpswrd=config['GitHub']['Password']
    g = Github(GHLogin, GHpswrd)

    ID_my=[]
    Parent_name=[]
    Parent_id=[]
    Parent_update=[]
    descr_my=[]
    Repo_source=[]
    created_my=[]
    size_my=[]
    own_repo = "Own repo"

    for repo in g.get_user().get_repos(): # going throung my repos
        ID_my.append(repo.id)
        if repo.parent is None: # if the repo is mine (i.e. no parent)
            Parent_name.append(repo.name)
            Parent_id.append(repo.id)
            Parent_update.append(repo.updated_at)
            Repo_source.append("Own")
        else:
            Parent_name.append(repo.parent.full_name)
            Parent_id.append(repo.parent.id)
            Parent_update.append(repo.parent.updated_at)
            Repo_source.append("Forked")
        descr_my.append(repo.description)
        created_my.append(repo.created_at)
        size_my.append(repo.size)
        
    for repo in g.get_user().get_starred(): # going through starred repos
        ID_my.append(repo.id)
        Parent_name.append(repo.name)
        Parent_id.append(repo.id) # the same as ID_my
        Parent_update.append(repo.updated_at)
        Repo_source.append("Star")
        descr_my.append(repo.description)
        created_my.append(repo.created_at)
        size_my.append(repo.size)
        
    my_repos = pd.DataFrame({'My_ID': ID_my,
                            'Source': Repo_source,
                            'Parent': Parent_name,
                            'Parent_ID': Parent_id,
                            'Parent_Update': Parent_update,
                            'Created': created_my,
                            'Size': size_my,
                            'Descrip': descr_my,
                            })

    my_repos.to_csv("Repos.csv")

#-------- Searching Github --------------------------------------------------------------

        from df2gspread import gspread2df as g2d

        from df2gspread import df2gspread as d2g

        d2g.upload(pr_instr, gfile='/Trading FXCM/PyData', wks_name='pr_instr')
        weights = g2d.download(gfile="1bmy2DLu5NV5IP-mo9rGWOyHOx7bEfoglVZmzzuHi5zc", wks_name="Weights", col_names=True, row_names=True, credentials=None, start_cell='A1')

        keywords_all = "IEX, hedge, oanda, quandl, NYSE, ETF, " \
                    "market calendar, equity, kelly, arbitrage, backtest, " \
                    "quant, EDGAR, del Prado, zorro trading"
        keywords = [keyword.strip() for keyword in keywords_all.split(',')]

        # exclude_keywords = "ng-zorro, ngx-zorro, ngzorro, CSS, Typescript"
        # exclude_keyword = [keyword.strip() for keyword in exclude_keywords.split(',')]
        # query = '+'.join(keyword) + '+NOT'+ '+'.join(exclude_keyword)+'pushed:>=2019-07-05'+'language:python' +'+in:readme+in:description'
        # result = g.search_repositories(query, 'stars', 'desc')


        ID_my=[]
        Parent_name=[]
        Parent_id=[]
        Parent_update=[]
        descr_my=[]
        Repo_source=[]
        created_my=[]

        for keyword in keywords:
            repositories = g.search_repositories(query=keyword+' language:python in:name in:readme in:description pushed:>2020-05-21', sort='updated')
            print(f'Found {repositories.totalCount} repo(s) for key={keyword}')
            for repo in repositories:
                ID_my.append(repo.id)
                Parent_name.append(repo.name)
                Parent_id.append(repo.id)
                Parent_update.append(repo.updated_at)
                created_my.append(repo.created_at)
                descr_my.append(repo.description)

        Search_result = pd.DataFrame({'My_ID': ID_my,
                                    'Parent': Parent_name,
                                    'Parent_ID': Parent_id,
                                    'Parent_Update': Parent_update,
                                    'Created': created_my,
                                    'Descrip': descr_my,
                                    })
        Search_result.to_csv("GH_Search_Results.csv")
