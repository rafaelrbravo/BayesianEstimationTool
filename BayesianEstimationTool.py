import numpyro as npo
import numpyro.distributions as dist
from jax import random, vmap
import jax.numpy as jnp
from pprint import pprint
import jax
import gc
import functools
import numpy as np
import inspect
import os
import warnings
from pathlib import Path

class Estimator():
    _samplePrefixes=("__globalParam__","__localParam__","__cholesky__")
    _modelOutputPrefix="__modelOutput__"

    def __init__(self,modelDataFull,ModelFn,localParamNames=None,GlobalParamFn=None,choleskyConcentration=2.0,rngSeed=None):
        self._ValidateInit(modelDataFull,ModelFn,localParamNames,GlobalParamFn,choleskyConcentration,rngSeed)
        self._modelDataFull=modelDataFull
        self._localParamNames=[] if localParamNames is None else localParamNames
        self._ModelFn=ModelFn
        self._GlobalFn=GlobalParamFn
        self._choleskyConcentration=choleskyConcentration
        self._modelFnName=self._GetFnName(self._ModelFn)
        self._globalFnName=self._GetFnName(self._GlobalFn)
        self._globalNames=None
        self._key=None if rngSeed is None else random.PRNGKey(rngSeed)
        self._ClearTrainState()
        self._ClearTestState()

    def Train(self,trainIndices,TrainLikelihoodFn,Method,printSummary=False,rngSeed=None,saveLocals=True,saveModelOut=False):
        self._ValidateTrain(trainIndices,TrainLikelihoodFn,Method,printSummary,rngSeed,saveLocals,saveModelOut)
        self._SetTrainState(trainIndices,TrainLikelihoodFn,Method)
        dataShapes={key:value.shape[1:] for key,value in self._modelDataFull.items()}
        fitModel=functools.partial(self._RunMethodTrain,TrainLikelihoodFn,dataShapes,saveModelOut,**self._GetTrainData())
        samplePrefixes=self._samplePrefixes+(self._modelOutputPrefix,) if saveModelOut else self._samplePrefixes
        result=Method._Fit(fitModel,self._NextKey(rngSeed),samplePrefixes)
        post=result["samples"]
        self._trainGlobals=self._GetPost(post,"__globalParam__")
        self._trainCholesky=post.get("__cholesky__")
        if saveLocals: self._trainLocals=self._GetPost(post,"__localParam__")
        if saveModelOut: self._trainModelOut=self._GetPost(post,"__modelOutput__")
        self._ClearTestState()
        if printSummary: self._PrintMethodSummary(Method,result["output"],post,self._localParamNames,self._globalNames,"--= Training posterior summary =--")
        jax.block_until_ready(post)
        self._ClearJAX()
        return result["output"]

    def Test(self,testIndices,TestLikelihoodFn=None,Method=None,nGlobalSamples=None,nLocalSamples=2000,printSummary=False,rngSeed=None,saveModelOut=False):
        self._ValidateTest(testIndices,TestLikelihoodFn,Method,nGlobalSamples,nLocalSamples,printSummary,rngSeed,saveModelOut)
        self._SetTestState(testIndices,TestLikelihoodFn,Method,nGlobalSamples,nLocalSamples)
        if TestLikelihoodFn is None:
            result=self._TestNoEstimator(nGlobalSamples,nLocalSamples,rngSeed,saveModelOut)
        else:
            fitResults=self._TestMethod(TestLikelihoodFn,Method,nGlobalSamples,rngSeed,saveModelOut)
            if printSummary:
                for i,fitResult in enumerate(fitResults):
                    title="--= Testing posterior summary =--" if len(fitResults)==1 else f"--= Testing posterior summary {i+1} =--"
                    if self._testGlobals:
                        print(title)
                        print("Global parameter values:")
                        for name,value in self._testGlobals.items(): print(f"  {name}: {np.asarray(value[i])}")
                        title=None
                    self._PrintMethodSummary(Method,fitResult["output"],fitResult["samples"],self._localParamNames,[],title)
            result=[fitResult["output"] for fitResult in fitResults]
        return result

    def ScoreTrain(self,ScoreFn,nSamples=None,rngSeed=None):
        self._ValidateScoreTrain(ScoreFn,nSamples,rngSeed)
        nPost=self._GetNTrainSamples()
        if nSamples is None: idxs=jnp.arange(nPost)
        else: idxs=self._GetRandomIndices( nSamples, nPost, self._NextKey(rngSeed))
        data=self._GetTrainData()
        if self._trainModelOut is not None: modelOut={ name:value[idxs] for name,value in self._trainModelOut.items() }
        else:
            globalParams={ name:value[idxs] for name,value in self._trainGlobals.items() }
            if self._localParamNames:
                localParams={ name:value[idxs] for name,value in self._trainLocals.items() }
                modelOut=vmap( vmap(self._ModelFn,in_axes=(None,0,0)), in_axes=(0,0,None))(globalParams,localParams,data)
            else: modelOut=vmap( vmap(self._ModelFn,in_axes=(None,None,0)), in_axes=(0,None,None))(globalParams,{},data)
        return vmap( vmap(ScoreFn,in_axes=(0,0)), in_axes=(0,None))(modelOut,data)

    def ScoreTest(self,ScoreFn,nSamples=None,rngSeed=None):
        self._ValidateScoreTest(ScoreFn,nSamples,rngSeed)
        if self._testModelOut is not None: nLocal=next(iter(self._testModelOut.values())).shape[1]
        else: nLocal=next(iter(self._testLocals.values())).shape[1]
        if nSamples is None: idxs=jnp.arange(nLocal)
        else: idxs=self._GetRandomIndices( nSamples, nLocal, self._NextKey(rngSeed))
        data=self._GetTestData()
        if self._testModelOut is not None: modelOut={ name:value[:,idxs] for name,value in self._testModelOut.items() }
        else:
            localParams={ name:value[:,idxs] for name,value in self._testLocals.items() }
            modelOut=self._RunModelSamples( self._testGlobals, localParams, data)
        return vmap( vmap( vmap(ScoreFn,in_axes=(0,0)), in_axes=(0,None)), in_axes=(0,None))(modelOut,data)

    def Save( self, fileName, saveTrainLocals=True, saveTestLocals=True, saveTrainModelOut=False, saveTestModelOut=False, saveAllData=False):
        self._ValidateSave(fileName,saveTrainLocals,saveTestLocals,saveTrainModelOut,saveTestModelOut,saveAllData)
        if fileName is None: return self._SaveRecord(saveTrainLocals,saveTestLocals,saveTrainModelOut,saveTestModelOut,saveAllData,flatten=False)
        record=self._SaveRecord(saveTrainLocals,saveTestLocals,saveTrainModelOut,saveTestModelOut,saveAllData,flatten=True)
        np.savez_compressed(fileName,**record)

    @classmethod
    def Load(cls, fileName, ModelFn, GlobalParamFn=None, modelDataFull=None):
        cls._ValidateLoad(fileName,ModelFn,GlobalParamFn,modelDataFull)
        fileName = Path(fileName)
        if fileName.suffix.lower() != ".npz": fileName = fileName.with_name(fileName.name + ".npz")
        if ModelFn is None: return cls._LoadToDict(fileName)
        return cls._Load(fileName,ModelFn,GlobalParamFn,modelDataFull)

    def _PrintMethodSummary(self,Method,output,samples,localNames,globalNames,title=None):
        mode=getattr(Method,"_printMode",None)
        if mode in ("before","replace"): Method._Print(output,localNames,globalNames,title)
        if mode!="replace":
            self._PrintSampleSummary(samples,localNames,globalNames,title if mode!="before" else None)
        if mode=="after": Method._Print(output,localNames,globalNames,None)

    def _PrintSampleSummary(self,samples,localNames,globalNames,title=None):
        if title is not None: print(title)
        print(f"{'parameter':<20} {'mean':>10} {'std':>10} {'median':>10} {'2.5%':>10} {'97.5%':>10}")
        for prefix,names in (("__globalParam__",[] if globalNames is None else list(globalNames)),("__localParam__",list(localNames))):
            for name in names:
                sampleName=f"{prefix}{name}"
                if sampleName not in samples: continue
                vals=np.asarray(samples[sampleName])
                shape=vals.shape[1:]
                if not shape: self._PrintSampleRow(name,vals)
                else:
                    for inds in np.ndindex(shape): self._PrintSampleRow(f"{name}[{','.join(map(str,inds))}]",vals[(slice(None),)+inds])

    def _PrintSampleRow(self,label,vals):
        print(f"{label:<20} {np.mean(vals):>10.3f} {np.std(vals):>10.3f} {np.median(vals):>10.3f} {np.percentile(vals,2.5):>10.3f} {np.percentile(vals,97.5):>10.3f}")

    def _TestMethod(self,TestLikelihoodFn,Method,nGlobalSamples=None,rngSeed=None,saveModelOut=False):
        postKey,runKey=random.split(self._NextKey(rngSeed),2)
        self._SetTestGlobals(nGlobalSamples,postKey)
        testData=self._GetTestData()
        nGlobalSamples=1 if nGlobalSamples is None else nGlobalSamples
        runKeys=random.split(runKey,nGlobalSamples)
        results=[]
        localPosts=[]
        modelOutPosts=[]
        for i in range(nGlobalSamples):
            globs={name:value[i] for name,value in self._testGlobals.items()}
            cholesky=None if self._testCholesky is None else self._testCholesky[i]
            fitModel=functools.partial(self._RunMethodTest,TestLikelihoodFn,globs,cholesky,saveModelOut,**testData)
            samplePrefixes=("__localParam__",self._modelOutputPrefix) if saveModelOut else ("__localParam__",)
            result=Method._Fit(fitModel,runKeys[i],samplePrefixes)
            localPosts.append(self._GetPost(result["samples"],"__localParam__"))
            if saveModelOut: modelOutPosts.append(self._GetPost(result["samples"],"__modelOutput__"))
            results.append(result)
        self._testLocals={} if not localPosts or not localPosts[0] else {name:jnp.stack([p[name] for p in localPosts]) for name in localPosts[0]}
        if saveModelOut: self._testModelOut={} if not modelOutPosts or not modelOutPosts[0] else {name:jnp.stack([p[name] for p in modelOutPosts]) for name in modelOutPosts[0]}
        jax.block_until_ready((self._testLocals,self._testModelOut))
        self._ClearJAX()
        return results

    def _TestNoEstimator(self,nGlobalSamples,nLocalSamples,rngSeed=None,saveModelOut=False):
        postKey,sampleKey=random.split(self._NextKey(rngSeed),2)
        self._SetTestGlobals(nGlobalSamples,postKey)
        nGlobalSamples=1 if nGlobalSamples is None else nGlobalSamples
        testData=self._GetTestData()
        nTest=len(testData[next(iter(testData))])
        nLocal=len(self._localParamNames)
        if nLocal>0:
            localSamples=random.normal(sampleKey,(nGlobalSamples,nLocalSamples,nTest,nLocal))
            if nLocal>1:
                localSamples=jnp.einsum("gspi,gji->gspj",localSamples,self._testCholesky)
            localParams={name:localSamples[:,:,:,i] for i,name in enumerate(self._localParamNames)}
        else: localParams={}
        self._testLocals=localParams
        if saveModelOut:
            self._testModelOut=self._RunModelSamples(self._testGlobals,self._testLocals,testData)

    def _RunModelSamples(self,globalParams,localParams,data):
        return vmap( vmap( vmap(self._ModelFn,in_axes=(None,0,0)),in_axes=(None,0,None)),in_axes=(0,0,None))(globalParams,localParams,data)

    def _GetRandomIndices(self,nSamples,length,RNGkey):
        return random.choice(RNGkey,length,shape=(nSamples,),replace=nSamples>length)

    def _GenLocalParamsTraining(self,dataSize):
        nLocal=len(self._localParamNames)
        localSamples=npo.sample("__localSample__",dist.Normal(0.0,1.0).expand((dataSize,nLocal)))
        if nLocal>1:
            choleskySample=npo.sample("__choleskySample__",dist.LKJCholesky(nLocal,concentration=self._choleskyConcentration))
            cholesky=npo.deterministic("__cholesky__",choleskySample)
            localSamples=localSamples@cholesky.T
        return dict(zip(self._localParamNames,localSamples.T))

    def _RunMethodTrain(self,TrainLikelihoodFn,dataShapes,saveOutput,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        localParams=self._GenLocalParamsTraining(dataLen) if self._localParamNames else {}
        for name, value in localParams.items(): npo.deterministic(f"__localParam__{name}", value)
        globalParams=self._GlobalFn(dataShapes) if self._GlobalFn is not None else {}
        if self._globalNames is None: self._globalNames=list(globalParams)
        for name,value in globalParams.items(): npo.deterministic(f"__globalParam__{name}",value)
        modelOut=vmap(self._ModelFn,in_axes=(None,0,0))(globalParams,localParams,modelData)
        if saveOutput:
            for name,value in modelOut.items(): npo.deterministic(f"__modelOutput__{name}",value)
        TrainLikelihoodFn(globalParams,localParams,modelData,modelOut)

    def _GenLocalParamsTesting(self,cholesky,dataSize):
        nLocal=len(self._localParamNames)
        localSamples=npo.sample("__localSamples__",dist.Normal(0.0,1.0).expand((dataSize,nLocal)))
        if nLocal>1:
            localSamples=localSamples @ cholesky.T
        return dict(zip(self._localParamNames,localSamples.T))

    def _RunMethodTest(self,TestLikelihoodFn,globalParams,cholesky,saveOutput,**modelData):
        dataLen=len(modelData[next(iter(modelData))])
        localParams=self._GenLocalParamsTesting(cholesky,dataLen) if self._localParamNames else {}
        for name,value in localParams.items(): npo.deterministic(f"__localParam__{name}",value)
        modelOut=vmap(self._ModelFn,in_axes=(None,0,0))(globalParams,localParams,modelData)
        if saveOutput:
            for name,value in modelOut.items(): npo.deterministic(f"__modelOutput__{name}",value)
        TestLikelihoodFn(globalParams,localParams,modelData,modelOut)

    def _NextKey(self,rngSeed=None):
        if rngSeed is not None:
            self._key=random.PRNGKey(rngSeed)
        self._key,key=random.split(self._key)
        return key

    def _GetTrainData(self):
        return { k: v[self._trainIndices] for k, v in self._modelDataFull.items() }

    def _GetTestData(self):
        return { k: v[self._testIndices] for k, v in self._modelDataFull.items() }

    def _GetNTrainSamples(self):
        return 0 if not self._trainGlobals else len(next(iter(self._trainGlobals.values())))

    def _GetFnName(self,Fn):
        if Fn is None: return None
        if isinstance(Fn, functools.partial): return Fn.func.__name__
        return getattr(Fn, "__name__", type(Fn).__name__)

    def _GetPost(self,post, prefix): 
        ret = { name.removeprefix(prefix): value for name, value in post.items() if name.startswith(prefix) }
        if not ret: return {}
        return ret


    def _SaveRecord(self,saveTrainLocals=True,saveTestLocals=True,saveTrainModelOut=False,saveTestModelOut=False,saveAllData=False,flatten=True):
        record={}
        self._AddInfo(record,"meta",flatten,{
            "localParamNames":np.asarray(self._localParamNames,dtype=str),
            "globalNames":None if self._globalNames is None else np.asarray(self._globalNames,dtype=str),
            "choleskyConcentration":self._choleskyConcentration,
            "modelFnName":self._modelFnName,
            "globalFnName":self._globalFnName})
        self._AddInfo(record,"state",flatten,{"trainIndices":self._trainIndices,"testIndices":self._testIndices,"rngKey":self._key})
        self._AddInfo(record,"trainParams",flatten,self._trainParams)
        self._AddInfo(record,"testParams",flatten,self._testParams)
        self._AddInfo(record,"trainGlobals",flatten,self._trainGlobals)
        if self._trainCholesky is not None: record["trainCholesky"]=np.asarray(self._trainCholesky)
        if saveTrainLocals: self._AddInfo(record,"trainLocals",flatten,self._trainLocals)
        if saveTestLocals and self._testLocals:
            self._AddInfo(record,"testGlobals",flatten,self._testGlobals)
            self._AddInfo(record,"testLocals",flatten,self._testLocals)
            if self._testCholesky is not None: record["testCholesky"]=np.asarray(self._testCholesky)
        if saveTrainModelOut and self._trainModelOut is not None: self._AddInfo(record,"trainModelOut",flatten,self._trainModelOut)
        if saveTestModelOut and self._testModelOut is not None: self._AddInfo(record,"testModelOut",flatten,self._testModelOut)
        if saveAllData: self._AddInfo(record,"modelDataFull",flatten,self._modelDataFull)
        return record

    @classmethod
    def _Load(cls,fileName,ModelFn,GlobalParamFn=None,modelDataFull=None):
        with np.load(fileName,allow_pickle=False) as record:
            meta=cls._GetInfo(record,"meta")
            state=cls._GetInfo(record,"state")
            savedData=cls._GetInfo(record,"modelDataFull")
            if modelDataFull is None:
                if not savedData: raise ValueError("modelDataFull must be provided because the saved framework does not contain the original model data.")
                modelDataFull=savedData
            localParamNames=meta.get("localParamNames")
            localParamNames=[] if localParamNames is None else list(localParamNames)
            framework=cls(modelDataFull,ModelFn,localParamNames=localParamNames,GlobalParamFn=GlobalParamFn,choleskyConcentration=meta["choleskyConcentration"])
            globalNames=meta.get("globalNames")
            framework._globalNames=None if globalNames is None else list(globalNames)
            framework._trainParams=cls._GetInfo(record,"trainParams") or None
            framework._testParams=cls._GetInfo(record,"testParams") or None
            framework._trainGlobals={name:jnp.asarray(value) for name,value in cls._GetInfo(record,"trainGlobals").items()}
            framework._trainLocals={name:jnp.asarray(value) for name,value in cls._GetInfo(record,"trainLocals").items()}
            framework._trainCholesky=jnp.asarray(record["trainCholesky"]) if "trainCholesky" in record.files else None
            framework._testGlobals={name:jnp.asarray(value) for name,value in cls._GetInfo(record,"testGlobals").items()}
            framework._testLocals={name:jnp.asarray(value) for name,value in cls._GetInfo(record,"testLocals").items()}
            framework._testCholesky=jnp.asarray(record["testCholesky"]) if "testCholesky" in record.files else None
            trainModelOut=cls._GetInfo(record,"trainModelOut")
            testModelOut=cls._GetInfo(record,"testModelOut")
            framework._trainModelOut={name:jnp.asarray(value) for name,value in trainModelOut.items()} if trainModelOut else None
            framework._testModelOut={name:jnp.asarray(value) for name,value in testModelOut.items()} if testModelOut else None
            framework._trainIndices=None if state.get("trainIndices") is None else np.asarray(state["trainIndices"])
            framework._testIndices=None if state.get("testIndices") is None else np.asarray(state["testIndices"])
            framework._key=None if state.get("rngKey") is None else jnp.asarray(state["rngKey"])
            savedModelFnName=meta.get("modelFnName")
            currentModelFnName=framework._GetFnName(ModelFn)
            if savedModelFnName!=currentModelFnName: warnings.warn(f"Saved ModelFn was '{savedModelFnName}', but the supplied ModelFn is '{currentModelFnName}'.",RuntimeWarning)
            savedGlobalFnName=meta.get("globalFnName")
            currentGlobalFnName=framework._GetFnName(GlobalParamFn)
            if savedGlobalFnName!=currentGlobalFnName: warnings.warn(f"Saved GlobalParamFn was '{savedGlobalFnName}', but the supplied GlobalParamFn is '{currentGlobalFnName}'.",RuntimeWarning)
        return framework

    @staticmethod
    def _LoadToDict(path):
        with np.load(path,allow_pickle=False) as flat:
            record={}
            for key,value in flat.items():
                parts=key.split("/")
                if len(parts)==1:
                    record[key]=value.item() if value.ndim==0 else value
                    continue
                if len(parts) not in (2,3): raise ValueError(f"Invalid saved key '{key}'.")
                prefix,name=parts[:2]
                if not prefix or not name: raise ValueError(f"Invalid saved key '{key}'.")
                if len(parts)==3 and parts[2]!="__none__": raise ValueError(f"Invalid saved key '{key}'.")
                if prefix not in record: record[prefix]={}
                record[prefix][name]=None if len(parts)==3 else (value.item() if value.ndim==0 else value)
        return record

    @staticmethod
    def _AddInfo(record,prefix,flatten,info):
        if info is None: return
        if not flatten:
            record[prefix]=info
            return
        for name,value in info.items():
            if not isinstance(name,str): raise TypeError(f"Saved dictionary keys must be strings; got {type(name).__name__}.")
            if not name: raise ValueError("Saved dictionary keys cannot be empty.")
            if "/" in name: raise ValueError(f"Saved dictionary key '{name}' cannot contain '/'.")
            if value is None: record[f"{prefix}/{name}/__none__"]=np.asarray(True)
            elif isinstance(value,dict):
                for subName,subValue in value.items():
                    if not isinstance(subName,str) or not subName or "/" in subName: raise ValueError(f"Invalid nested saved key '{subName}'.")
                    if subValue is None: record[f"{prefix}/{name}.{subName}/__none__"]=np.asarray(True)
                    else: record[f"{prefix}/{name}.{subName}"]=np.asarray(subValue)
            else: record[f"{prefix}/{name}"]=np.asarray(value)

    @staticmethod
    def _GetInfo(record,prefix):
        info={}
        start=f"{prefix}/"
        noneSuffix="/__none__"
        keys=record.files if hasattr(record,"files") else record.keys()
        for key in keys:
            if not key.startswith(start): continue
            name=key.removeprefix(start)
            isNone=name.endswith(noneSuffix)
            if isNone: name=name.removesuffix(noneSuffix)
            if not name or "/" in name: raise ValueError(f"Invalid saved key '{key}'.")
            value=None if isNone else record[key]
            value=value if value is None or value.ndim else value.item()
            if "." in name:
                parent,child=name.split(".",1)
                info.setdefault(parent,{})[child]=value
            else: info[name]=value
        return info

    def _ValidateInit(self,modelDataFull,ModelFn,localParamNames,GlobalParamFn,choleskyConcentration,rngSeed):
        self._CheckModelData(modelDataFull)
        self._CheckCallable(ModelFn,"ModelFn",3)
        self._CheckCallable(GlobalParamFn,"GlobalParamFn",1,optional=True)
        if localParamNames is not None:
            if not isinstance(localParamNames,(list,tuple)): raise TypeError("localParamNames must be a list or tuple of strings, or None.")
            for i,name in enumerate(localParamNames):
                if not isinstance(name,str): raise TypeError(f"localParamNames[{i}] must be a string; got {type(name).__name__}.")
                if not name: raise ValueError(f"localParamNames[{i}] cannot be empty.")
            if len(set(localParamNames))!=len(localParamNames): raise ValueError("localParamNames cannot contain duplicate names.")
        try: concentration=np.asarray(choleskyConcentration)
        except Exception as e: raise TypeError("choleskyConcentration must be a positive finite real scalar.") from e
        if concentration.ndim!=0: raise TypeError("choleskyConcentration must be a scalar.")
        if not np.issubdtype(concentration.dtype,np.number) or np.issubdtype(concentration.dtype,np.complexfloating): raise TypeError("choleskyConcentration must be a real numeric scalar.")
        concentration=concentration.item()
        if not np.isfinite(concentration): raise ValueError("choleskyConcentration must be finite.")
        if concentration<=0: raise ValueError("choleskyConcentration must be > 0.")
        if rngSeed is not None: self._CheckNonnegativeInt(rngSeed,"RNG seed")

    def _ValidateTrain(self,trainIndices,TrainLikelihoodFn,Method,printSummary,rngSeed,saveLocals,saveModelOut):
        self._CheckIndices(trainIndices,len(self._modelDataFull[next(iter(self._modelDataFull))]),"trainIndices")
        self._CheckCallable(TrainLikelihoodFn,"TrainLikelihoodFn",4)
        self._CheckMethod(Method,"Method")
        self._CheckBool(printSummary,"printSummary"); self._CheckBool(saveLocals,"saveLocals"); self._CheckBool(saveModelOut,"saveModelOut")
        self._CheckRngAvailable(rngSeed)

    def _ValidateTest(self,testIndices,TestLikelihoodFn,Method,nGlobalSamples,nLocalSamples,printSummary,rngSeed,saveModelOut):
        if self._trainParams is None: raise RuntimeError("Train() must be run before Test().")
        if not self._localParamNames: raise RuntimeError("Test() requires at least one local parameter.")
        self._CheckIndices(testIndices,len(self._modelDataFull[next(iter(self._modelDataFull))]),"testIndices")
        self._CheckCallable(TestLikelihoodFn,"TestLikelihoodFn",4,optional=True)
        if nGlobalSamples is not None: self._CheckPositiveInt(nGlobalSamples,"nGlobalSamples")
        if TestLikelihoodFn is None:
            if Method is not None: raise ValueError("Method must be None when TestLikelihoodFn is None.")
            self._CheckPositiveInt(nLocalSamples,"nLocalSamples")
        else: self._CheckMethod(Method,"Method")
        self._CheckBool(printSummary,"printSummary"); self._CheckBool(saveModelOut,"saveModelOut")
        self._CheckRngAvailable(rngSeed)

    def _ValidateScoreTrain(self,ScoreFn,nSamples,rngSeed):
        if self._trainParams is None: raise RuntimeError("Train() must be run before ScoreTrain().")
        self._CheckCallable(ScoreFn,"ScoreFn",2)
        if nSamples is not None:
            self._CheckPositiveInt(nSamples,"nSamples"); self._CheckRngAvailable(rngSeed)
        if self._trainModelOut is None and self._localParamNames and not self._trainLocals: raise RuntimeError("ScoreTrain() requires saved training locals or saved training model outputs when the model has local parameters.")

    def _ValidateScoreTest(self,ScoreFn,nSamples,rngSeed):
        if self._testParams is None: raise RuntimeError("Test() must be run before ScoreTest().")
        if self._testModelOut is None and not self._testLocals: raise RuntimeError("ScoreTest() requires saved test locals or saved test model outputs.")
        self._CheckCallable(ScoreFn,"ScoreFn",2)
        if nSamples is not None:
            self._CheckPositiveInt(nSamples,"nSamples"); self._CheckRngAvailable(rngSeed)

    def _ValidateSave(self,fileName,saveTrainLocals,saveTestLocals,saveTrainModelOut,saveTestModelOut,saveAllData):
        if fileName is not None and not isinstance(fileName,(str,os.PathLike)): raise TypeError("fileName must be a string, path-like object, or None.")
        for value,name in ((saveTrainLocals,"saveTrainLocals"),(saveTestLocals,"saveTestLocals"),(saveTrainModelOut,"saveTrainModelOut"),(saveTestModelOut,"saveTestModelOut"),(saveAllData,"saveAllData")): self._CheckBool(value,name)

    @classmethod
    def _ValidateLoad(cls,fileName,ModelFn,GlobalParamFn,modelDataFull):
        if not isinstance(fileName,(str,os.PathLike)): raise TypeError("fileName must be a string or path-like object.")
        cls._CheckCallable(ModelFn,"ModelFn",3,optional=True)
        cls._CheckCallable(GlobalParamFn,"GlobalParamFn",1,optional=True)
        if modelDataFull is not None: cls._CheckModelData(modelDataFull)

    def _SetTrainState(self,trainIndices,TrainLikelihoodFn,Method):
        self._trainParams={"trainLikelihoodName":self._GetFnName(TrainLikelihoodFn),"methodName":type(Method).__name__,"methodArgs":Method._SaveArgs()}
        self._trainIndices=trainIndices
        self._globalNames=None
        self._trainGlobals={}
        self._trainLocals={}
        self._trainCholesky=None
        self._trainModelOut=None

    def _SetTestState(self,testIndices,TestLikelihoodFn,Method,nGlobalSamples,nLocalSamples):
        self._testParams={"mode":"NoMethod" if TestLikelihoodFn is None else "Method","nGlobalSamples":nGlobalSamples,"nLocalSamples":nLocalSamples if TestLikelihoodFn is None else None,"testLikelihoodName":self._GetFnName(TestLikelihoodFn),"methodName":None if Method is None else type(Method).__name__,"methodArgs":None if Method is None else Method._SaveArgs()}
        self._testIndices=testIndices
        self._testGlobals={}
        self._testLocals={}
        self._testCholesky=None
        self._testModelOut=None

    def _SetTestGlobals(self,nGlobalSamples,RNGkey):
        if nGlobalSamples is None:
            self._testGlobals={name:jnp.expand_dims(jnp.median(value,axis=0),axis=0) for name,value in self._trainGlobals.items()}
            self._testCholesky=None if self._trainCholesky is None else jnp.expand_dims(jnp.median(self._trainCholesky,axis=0), axis=0)
        else:
            nTrainSamples=self._GetNTrainSamples()
            idxs=self._GetRandomIndices(nGlobalSamples,nTrainSamples,RNGkey)
            self._testGlobals={ name:value[idxs] for name,value in self._trainGlobals.items() }
            self._testCholesky=None if self._trainCholesky is None else self._trainCholesky[idxs]

    @staticmethod
    def _CheckBool(value,name):
        if not isinstance(value,(bool,np.bool_)): raise TypeError(f"{name} must be bool; got {type(value).__name__}.")

    @staticmethod
    def _CheckPositiveInt(value,name):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)): raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
        if value<=0: raise ValueError(f"{name} must be > 0.")

    @staticmethod
    def _CheckNonnegativeInt(value,name):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)): raise TypeError(f"{name} must be an integer; got {type(value).__name__}.")
        if value<0: raise ValueError(f"{name} must be >= 0.")

    @staticmethod
    def _CheckCallable(fn,name,nArgs,optional=False):
        if fn is None:
            if optional: return
            raise TypeError(f"{name} cannot be None.")
        if not callable(fn): raise TypeError(f"{name} must be callable; got {type(fn).__name__}.")
        try: signature=inspect.signature(fn)
        except (TypeError,ValueError): return
        try: signature.bind(*([None]*nArgs))
        except TypeError as e: raise TypeError(f"{name} must accept {nArgs} positional argument{'s' if nArgs!=1 else ''}. Framework call will be incompatible with signature {signature}.") from e

    @staticmethod
    def _CheckMethod(estimator,name,optional=False):
        if estimator is None:
            if optional: return
            raise TypeError(f"{name} cannot be None.")
        for method in ("_Fit","_SaveArgs"):
            if not callable(getattr(estimator,method,None)): raise TypeError(f"{name} must implement callable {method}().")
        mode=getattr(estimator,"_printMode",None)
        if mode not in (None,"before","after","replace"): raise ValueError(f"{name}._printMode must be None, 'before', 'after', or 'replace'.")
        if mode is not None and not callable(getattr(estimator,"_Print",None)): raise TypeError(f"{name} must implement callable _Print() when _PrintMode is not None.")
        state=estimator._SaveArgs()
        if not isinstance(state,dict): raise TypeError(f"{name}._SaveArgs() must return a dictionary.")


    @staticmethod
    def _CheckModelData(modelDataFull):
        if not isinstance(modelDataFull,(dict,np.lib.npyio.NpzFile)): raise TypeError("modelDataFull must be a dictionary.")
        if len(modelDataFull)==0: raise ValueError("modelDataFull cannot be empty.")
        dataSize=None
        for name,value in modelDataFull.items():
            if not isinstance(name,str): raise TypeError(f"modelDataFull keys must be strings; got {type(name).__name__}.")
            if not name: raise ValueError("modelDataFull keys cannot be empty strings.")
            if not isinstance(value,(jax.Array,np.ndarray)): raise TypeError(f"modelDataFull['{name}'] must be a JAX or NumPy array; got {type(value).__name__}.")
            if value.ndim==0: raise ValueError(f"modelDataFull['{name}'] must have at least one dimension because its first dimension represents data entries.")
            if dataSize is None: dataSize=len(value)
            elif len(value)!=dataSize: raise ValueError(f"All arrays in modelDataFull must have the same first dimension. Expected {dataSize}, but modelDataFull['{name}'] has length {len(value)}.")
        if dataSize==0: raise ValueError("modelDataFull must contain at least one data entry.")

    @staticmethod
    def _CheckIndices(indices,dataSize,name):
        if not isinstance(indices,(list,tuple,np.ndarray,jax.Array)): raise TypeError(f"{name} must be a list, tuple, NumPy array, or JAX array.")
        arr=np.asarray(indices)
        if arr.ndim!=1: raise ValueError(f"{name} must be one-dimensional.")
        if len(arr)==0: raise ValueError(f"{name} cannot be empty.")
        if not np.issubdtype(arr.dtype,np.integer): raise TypeError(f"{name} must contain integers.")
        if np.any(arr<0): raise ValueError(f"{name} cannot contain negative indices.")
        if np.any(arr>=dataSize): raise IndexError(f"{name} contains an index outside modelDataFull, which contains {dataSize} entries.")
        if len(np.unique(arr))!=len(arr): raise ValueError(f"{name} cannot contain duplicate indices.")

    def _CheckRngAvailable(self,rngSeed):
        if rngSeed is not None:
            self._CheckNonnegativeInt(rngSeed,"RNG seed")
            return
        if self._key is None: raise ValueError("An RNG seed must be provided either when constructing the framework or for this call.")

    def _ClearTrainState(self):
        self._trainParams=None
        self._trainIndices=None
        self._trainGlobals={}
        self._trainLocals={}
        self._trainCholesky=None
        self._trainModelOut=None

    def _ClearTestState(self):
        self._testParams=None
        self._testIndices=None
        self._testGlobals={}
        self._testLocals={}
        self._testCholesky=None
        self._testModelOut=None

    def _ClearJAX(self):
        jax.clear_caches()
        gc.collect()
