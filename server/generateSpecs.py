import json
import pandas
import pandas as pd
import numpy as np


def generate_datatypes(data_df, columnOrder):
    dataset_fields = data_df.columns.tolist()
    dataset_types = []
    dataset_typelist = []
    labellist = []

    # force d3mIndex to be returned first
    dataset_typelist.append({'d3mIndex':'int64'})
    labellist.append('d3mIndex')
    print('generate datatypes: (order)',columnOrder)
    #for key in data_df.dtypes.to_dict():
    for key in columnOrder:
        print(key)
        if key != 'd3mIndex':
            columntype = str(data_df.dtypes.to_dict()[key])
            if columntype == 'object':
                columntype = "string"
            labellist.append(key)
            dataset_typelist.append({key: columntype})
    print('returning typelist,labels:')
    print(dataset_typelist,labellist)
    return (dataset_typelist,labellist)

# almost duplicate of the one above, just arranged the output differently for key/value 
# lookup instead of preserving order,the way the generate_datatypes does
def generate_datadictionary(data_df):
    dataset_fields = data_df.columns.tolist()
    dataset_types = []
    dataset_typelist = []
    labeldict = {}
    for key in data_df.dtypes.to_dict():
        columntype = str(data_df.dtypes.to_dict()[key])
        if columntype == 'object':
            columntype = "string"
            #TODO: need to test for integer values here instead of returning float64
        labeldict[key] = columntype
    return labeldict


def generate_dynamic_problem_spec(data_df,columnNameOrder,targetColumnName=None):
    problemSpec = {}
    performanceMetrics = []
    inputs = {}
    expectedOutputs = {}
    
    ###  ABOUT record
    about = {}
    about['problemID'] = 'dynamicProblem-ModSquad'
    about['problemName'] = 'dynamicProblem-ModSquad'
    about['problemSchemaVersion'] =  "3.1.1"
    about['problemDescription'] = 'dynamic problem'
    about['problemVersion'] = "1.0"
 
    (dtypes,labels) = generate_datatypes(data_df,columnNameOrder)   

    if targetColumnName != None:
        try:
            index = labels.index(targetColumnName)
        except:
            #couldn't find the lookup, so assume the last variable
            index = len(labels)-1
    else:
        index = len(labels)-1
        
    lastlabel = labels[index]
    lasttype_spec = dtypes[index]
    lasttype = lasttype_spec[lastlabel]
   
    
    # if the target is categorical (like a string), then assume classification, otherwise assume regression
    if lasttype in ['string', 'integer']:
        performanceMetrics = [{'metric': 'f1Macro'}]
        about['taskType'] = 'classification'
    else:
        about['taskType'] = 'regression'   
        #performanceMetrics = [{'metric': 'meanAbsoluteError'}] 
        performanceMetrics = [{'metric': 'rSquared'}] 
    
    # targets
    target = {}
    target['targetIndex'] = 0
    target['resID'] = "0"    # I don't remember what this is for
    print('labels:',labels)
    target['colIndex'] = index
    target['colName'] =  labels[index]
    targets = [target]
    
    # data record
    data = [{'datasetID':"dynamic-dataset-ModSquad",'targets': targets}]
    
    # dataSplits record
    dataSplits = {}
    dataSplits['method'] = 'holdOut'
    dataSplits['testSize'] = 0.20
    dataSplits['stratified'] = False
    dataSplits['numRepeats'] = 0
    
    # we don't write this file, but training might be ok
    dataSplits['splitsFile'] = 'dataSplits.csv'
    
    ### INPUTs record
    # inputs has  data, dataSplits, performanceMetrics
    inputs['performanceMetrics'] = performanceMetrics
    inputs['dataSplits'] = dataSplits
    inputs['data'] = data
   
    # compose the final record
    expectedOutputs = [{'predictionsFile': 'predictions.csv'}]
    problemSpec['about'] = about
    problemSpec['inputs'] = inputs
    problemSpec['expectedOutputs'] = expectedOutputs
    return problemSpec


# this is the form that the UI is expecting it, so just transform the full spec
def generateReturnedProblem(problemSpec):
    problems = []
    problems.append({'problemId': problemSpec['about']['problemID'],
                      'metrics' : problemSpec['inputs']['performanceMetrics'],
                      'targets':  problemSpec['inputs']['data'][0]['targets']
                    })
    return problems

# generate metadata the way the UI are pipelines are used to it
def generateMetadata(dataset_typelist,labels):
    metadata = []
    count = 0
    for entry in dataset_typelist:
        record = {}
        # each entry is { name : type}
        record['colIndex'] = count
        record['colName'] = labels[count]

        internalType = entry[labels[count]]
        if internalType == 'int64':
            record['colType'] = 'integer'
        elif internalType == 'string':
            record['colType'] = 'categorical'
        elif internalType == 'float64':
            record['colType'] = 'real'
        else:
            record['colType'] = 'unknown'
            print('found unmapped datatype:',internalType)

        # set the d3mIndex as the index variable, set the last variable as the target
        # TODO: find a way to pick the target instead of assuming the last column
        if record['colName'] == 'd3mIndex':
            record['role'] =  ['index']
        elif count == len(labels)-1:
            record['role'] = ['suggestedTarget']
        else:
            record['role'] = ['attribute']
        count += 1
        metadata.append(record)
    return metadata

# make the database spec record (materialized into databaseDoc.json)
# this will  the target var
def generate_database_spec(problemSpec,data_df, columnOrder):
    about = {}
    dataResources = []
    
    ###  ABOUT record
    about = {}
    about['datasetID'] = 'dynamicProblem-ModSquad'
    about['datasetName'] = 'dynamicProblem-ModSquad'
    about['citation'] = ''
    about['source'] = ''
    about['license'] = ''
    about["sourceURI"]= "https://www.acleddata.com/"
    about['redacted'] = False
    about['datasetSchemaVersion'] =  "3.0.0"
    about['datasetVersion'] = "1.0"
    # add the columns
    cols = []
    (types,labels) = generate_datatypes(data_df, columnOrder)
    metadata = generateMetadata(types,labels)
    # put in the declaration of where the data will be
    res = {}
    res['resID'] = 0
    res['resPath'] = 'tables/learningData.csv'
    res['resType'] = 'table'
    res['resFormat'] = ['text/csv']
    res['isCollection'] = False
    res['columns'] = metadata
    dataResources = [res]
    
    databaseSpec = {}
    databaseSpec['about'] = about
    databaseSpec['dataResources'] = dataResources
    return databaseSpec

def generateConfig():
    record = {}
    record["problem_schema"]= "/output/modsquad_files/problemDoc.json"
    record["problem_root"]= "/output/modsquad_files"
    record["dataset_schema"]= "/output/modsquad_files/datasetDoc.json"
    record["training_data_root"]= "/output/modsquad_files/"
    record["pipeline_logs_root"]= "/output/pipelines"
    record["executables_root"]= "/output/executables"
    record["temp_storage_root"]= "/output/supporting_files"
    record['dataset_schema'] = '/output/modsquad_files/datasetDoc.json' 
    record["dynamic_mode"] = True
    return record

def writeDatabaseDocFile(path,databaseSpec):
    filename = path+'/datasetDoc.json'
    print('write database file:',filename)
    with open(filename, 'w+') as outfile:
        json.dump(databaseSpec, outfile)

# It is possible the dataframe does not have a d3mIndex column yet.  
# If not, add one so TA2 can process the data and it looks like a usual d3m dataset.
def addIndexColumnIfNeeded(data_df):
    # the reindex call makes d3mIndex the first column in the dataset, which is a d3m convention
    if 'd3mIndex' not in data_df:
        data_df['d3mIndex'] = data_df.index
        data_df = reindex_dataframe_columns(dframe=data_df, columns=['d3mIndex'], new_indices=[0])
        #print('after reindexing:')
        #print(data_df)
    return data_df

def addIndexLabelIfNeeded(columnList):
    if 'd3mIndex' not in columnList:
        columnList = columnList.insert(0,'d3mIndex' )
    return columnList


#we need to write out the dataset so that TA2 can read it. 
def writeDatasetContents(path,databaseSpec,data_df):
    filename = path+'/'+databaseSpec['dataResources'][0]['resPath']
    data_df.to_csv(filename,index=False)

def readDatasetContents(path,databaseSpec):
    filename = path+'/'+databaseSpec['dataResources'][0]['resPath']
    data_df = pd.read_csv(filename, sep=',')
    return data_df

def writeConfig(path,configSpec):
    filename = path+'/search-config.json'
    with open(filename, 'w+') as outfile:
        json.dump(configSpec, outfile)
    filename = path+'/config.json'
    with open(filename, 'w+') as outfile:
        json.dump(configSpec, outfile)

def writeProblemSpecFile(path,problemSpec):
    filename = path+'/problemDoc.json'
    with open(filename, 'w+') as outfile:
        json.dump(problemSpec, outfile)

def readProblemSpecFile(path):
    filename = path+'/problemDoc.json'
    with open(filename) as specfile:
        return json.load(specfile)

def readDatasetDocFile(path):
    filename = path+'/datasetDoc.json'
    with open(filename) as specfile:
        return json.load(specfile)


# the user may have selected a different target than the one we used
# when the spec was created, so revise the target part and return the updated
# spec.

def updateSpecsForTarget(dynamic_problem_root,problem_spec,dataset_spec,targetName=None):
    # first find the column in the dataset spec that matches the desired target and find
    # the column currently marked in the dataset_spec as the target
    columnList = dataset_spec['dataResources'][0]['columns']
    try:
        previousTargetIndex = -1
        newTargetIndex = -1
        count = 0
        for entry in columnList:
            if entry['colName'] == targetName:
                newTargetIndex = count
            if entry['role'] == ['suggestedTarget']:
                previousTargetIndex = count
            count += 1
        print('target locations:',previousTargetIndex,newTargetIndex)
    except:
        print('error occurred while searching dataset spec')

    # we are changing the target and putting the new target last, so clear the previous
    # target label 
    columnList[previousTargetIndex]['role'] = ['attribute']

    # we want to remove the target declaration and insert it at the end of the list
    # since the FL TA2 seems to always need the target to be last.  This will update the
    # datasetDoc.json

    savedTargetRecord = columnList[newTargetIndex]
    savedTargetRecord['role'] = ['suggestedTarget']
    removed = columnList.pop(newTargetIndex)
    columnList.append(savedTargetRecord)

    # now relabel the colIndex values according to the new order in this list
    count = 0
    for entry in columnList:
        entry['colIndex'] = count
        count += 1


    # make new targets record for the problem spec
    target = {}
    target['targetIndex'] = 0
    target['resID'] = "0"    # I don't remember what this is for
    # we have modified the data so the target is always the last column
    target['colIndex'] = len(columnList)-1
    target['colName'] =  targetName
    targets = [target]
    # data record
    data = [{'datasetID':"dynamic-dataset-ModSquad",'targets': targets}]
       
    # update the problem spec with new target
    problem_spec['inputs']['data'] = data

    # fix the tasktype depending on the target type
    if savedTargetRecord['colType'] == 'categorical':
        problem_spec['about']['taskType'] = 'classification'
        problem_spec['inputs']['performanceMetrics'] = [{'metric': 'f1Macro'}] 
    else:
        problem_spec['about']['taskType'] = 'regression'   
        #problem_spec['inputs']['performanceMetrics'] = [{'metric': 'meanAbsoluteError'}] 
        problem_spec['inputs']['performanceMetrics'] = [{'metric': 'rSquared'}] 
   

    # fix the files on the disk to be ordered with target last and have the specs match
    #rewriteDatasetWithNewTarget(dynamic_problem_root,dataset_spec,target)
    writeDatabaseDocFile(dynamic_problem_root, dataset_spec)
    writeProblemSpecFile(dynamic_problem_root, problem_spec)
    return (problem_spec,dataset_spec)
  

# the user might pick a different column to solve for.  Some TA2s need the target column
# as the last column of the array, so rearrange the data in this way and rewrite the 
# problem and dataset specs

def rewriteDatasetWithNewTarget(dynamic_problem_root,dataset_spec,target):
    data_df = readDatasetContents(dynamic_problem_root,dataset_spec)
    # fix the dataset so the target is at the end 
    data_df = reindex_dataframe_columns(dframe=data_df, columns=['d3mIndex',target], new_indices=[0,-1])
    dataset_contents = data_df.to_dict('records')
    dataset_row_count = data_df.shape[0]
    dataset_column_count = data_df.shape[1]
    writeDatasetContents(dynamic_problem_root,dataset_spec,data_df)
   


# helpful function to allow moving any dataframe columns around, 
# thanks to pandas user @jmwoloso], see https://github.com/pandas-dev/pandas/issues/4588
#
## move 'Column_1' to the end, move 'Column_2' to the beginning
#df = reindex_dataframe_columns(dframe=df,
#                     columns=['Column_1', 'Column_2'], new_indices=[-1, 0])

def reindex_dataframe_columns(dframe=None, columns=None, new_indices=None):
    """
    Reorders the columns of a dataframe as specified by
    `reorder_indices`. Values of `columns` should align with their
    respective values in `new_indices`.

    `dframe`: pandas dataframe.

    `columns`: list,pandas.core.index.Index, or numpy array; columns to
    reindex.

    `reorder_indices`: list of integers or numpy array; indices
    corresponding to where each column should be inserted during
    re-indexing.
    """
    print("Re-indexing columns.")
    try:
        df = dframe.copy()

        # ensure parameters are of correct type and length
        assert isinstance(columns, (pd.core.index.Index,
                                    list,
                                    np.array)),\
        "`columns` must be of type `pandas.core.index.Index` or `list`"

        assert isinstance(new_indices,
                          list),\
        "`reorder_indices` must be of type `list`"

        assert len(columns) == len(new_indices),\
        "Length of `columns` and `reorder_indices` must be equal"

        # check for negative values in `new_indices`
        if any(idx < 0 for idx in new_indices):

            # get a list of the negative values
            negatives = [value for value
                         in new_indices
                         if value < 0]

            # find the index location for each negative value in
            # `new_indices`
            negative_idx_locations = [new_indices.index(negative)
                                      for negative in negatives]

            # zip the lists
            negative_zipped = list(zip(negative_idx_locations,
                                       negatives))

            # replace the negatives in `new_indices` with their
            # absolute position in the index
            for idx, negative in negative_zipped:
                new_indices[idx] = df.columns.get_loc(df.columns[
                                                          negative])

        # re-order the index now
        # get all columns
        all_columns = df.columns

        # drop the columns that need to be re-indexed
        all_columns = all_columns.drop(columns)

        # now re-insert them at the specified locations
        zipped_columns = list(zip(new_indices,
                                  columns))

        for idx, column in zipped_columns:
            all_columns = all_columns.insert(idx,
                                             column)
        # re-index the dataframe
        df = df.ix[:, all_columns]

        print("Successfully re-indexed dataframe.")

    except Exception as e:
        print(e)
        print("Could not re-index columns. Something went wrong.")

    return df

