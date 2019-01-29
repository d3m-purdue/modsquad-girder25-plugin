import json
import pandas

def generate_datatypes(data_df):
    dataset_fields = data_df.columns.tolist()
    dataset_types = []
    dataset_typelist = []
    labellist = []
    for key in data_df.dtypes.to_dict():
        columntype = str(data_df.dtypes.to_dict()[key])
        if columntype == 'object':
            columntype = "string"
            #TODO: need to test for integer values here instead of returning float64
        labellist.append(key)
        dataset_typelist.append({key: columntype})
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


def generate_dynamic_problem_spec(data_df,targetColumnName=None):
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
 
    (dtypes,labels) = generate_datatypes(data_df)   

    #if targetColumnName != None:
        # need to find a matching entry in the datatypes array
    	#dataTypeDict = generate_datadictionary(data_df)
    	#lastlabel = targetColumnName
    	#lasttype_spec = dataTypeDict[lastlabel]
    #else:
        # look at the last variable in the df as a placeholder target
    lastlabel = labels[-1:]
    #print('labels:',labels)
    #print(dtypes)
    #print(dtypes[-1:])
    #print(dtypes[-1:][0][lastlabel[0]])
    #print('dtypes:',dtypes)
    lasttype_spec = dtypes[-1:][0]
    lasttype = lasttype_spec[lastlabel[0]]
   
    
    # if the target is categorical (like a string), then assume classification, otherwise assume regression
    if lasttype in ['string', 'integer']:
        performanceMetrics = [{'metric': 'f1Macro'}]
        about['taskType'] = 'classification'
    else:
        about['taskType'] = 'regression'   
        performanceMetrics = [{'metric': 'meanAbsoluteError'}] 
    
    # targets
    target = {}
    target['targetIndex'] = 0
    target['resID'] = '0'    # I don't remember what this is for
    print('labels:',labels)
    print('lastlabel:',lastlabel)
    print('lastlabel[0]:',lastlabel[0])
    target['colIndex'] = len(labels)-1 
    target['colName'] =  labels[-1:][0] 
    targets = [target]
    
    # data record
    data = [{'datasetID':"dynamic-dataset-ModSuqad",'targets': targets}]
    
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

        if count == 0:
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
def generate_database_spec(problemSpec,data_df):
    about = {}
    dataResources = []
    
    # add the columns
    cols = []
    (types,labels) = generate_datatypes(data_df)
    metadata = generateMetadata(types,labels)
    # put in the declaration of where the data will be
    res = {}
    res['resID'] = 0
    res['resPath'] = 'learningData.csv'
    res['resType'] = 'table'
    res['resFormat'] = ['text/csv']
    res['isCollection'] = False
    res['columns'] = metadata
    dataResources = [res]
    
    databaseSpec = {}
    databaseSpec['about'] = about
    databaseSpec['dataResources'] = dataResources
    return databaseSpec


def writeDatabaseDocFile(path,databaseSpec):
    filename = path+'/datasetDoc.json'
    with open(filename, 'w') as outfile:
        json.dump(databaseSpec, outfile)


def writeDatasetContents(path,databaseSpec,data_df):
    filename = path+databaseSpec['dataResources'][0]['resPath']
    data_df.to_csv(filename)

def writeProblemSpecFile(path,problemSpec):
    filename = path+'/problemDoc.json'
    with open(filename, 'w') as outfile:
        json.dump(problemSpec, outfile)

def readProblemSpecFile(path):
    filename = path+'/problemDoc.json'
    with open(filename) as specfile:
        return json.load(specfile)

def readDatasetDocFile(path):
    filename = path+'/datasetDoc.json'
    with open(filename) as specfile:
        return json.load(specfile)



