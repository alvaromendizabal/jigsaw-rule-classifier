"""Import exactly one independently verified, previously executed public Git tree."""
import hashlib,json,lzma,os,pathlib,subprocess,tempfile
BASE='22126fa3a2ec07555996b8cbd8cfc106d29618c9'
TREE='ae71df3286f36e3096410c7e6c36c9e13bedf58b'
COMMIT='6b41e0626d7ebedbcdc09a220804b5518e6e265a'
PATCH_SHA='47ae15880c05d0fb9889a28b0903438fdce86c09977363c02cce508afaa08817'
RAW_COMMIT=('tree '+TREE+'\nparent '+BASE+'\nauthor Alvaro Mendizabal <108156083+alvaromendizabal@users.noreply.github.com> 1789085956 +0000\ncommitter Alvaro Mendizabal <108156083+alvaromendizabal@users.noreply.github.com> 1789085956 +0000\n\nPublish completed support-selection diagnostic in canonical notebooks\n').encode()
def run(*args,input=None,env=None):
 return subprocess.run(['git',*args],input=input,env=env,check=True,capture_output=True,timeout=45).stdout.strip().decode()
def main(patch_path):
 payload=pathlib.Path(patch_path).read_bytes()
 if hashlib.sha256(payload).hexdigest()!=PATCH_SHA: raise ValueError('Transport checksum mismatch')
 entries=json.loads(lzma.decompress(payload,memlimit=256*1024*1024))
 if len(entries)!=33: raise ValueError('Publication file count mismatch')
 with tempfile.TemporaryDirectory() as tmp:
  env={**os.environ,'GIT_INDEX_FILE':str(pathlib.Path(tmp)/'index')}
  run('read-tree',BASE,env=env)
  for name,patch in entries.items():
   path=pathlib.PurePosixPath(name)
   if path.is_absolute() or '..' in path.parts or '.git' in path.parts: raise ValueError('Unsafe path')
   if patch['old_sha256'] is None: original=b''
   else: original=subprocess.run(['git','show',BASE+':'+name],check=True,capture_output=True,timeout=10).stdout
   if patch['old_sha256'] is not None and hashlib.sha256(original).hexdigest()!=patch['old_sha256']: raise ValueError('Base bytes differ: '+name)
   lines=original.decode().splitlines(keepends=True)
   data=''.join(''.join(lines[p[0]:p[1]]) if isinstance(p,list) else p for p in patch['pieces']).encode()
   if hashlib.sha256(data).hexdigest()!=patch['sha256']: raise ValueError('Candidate bytes differ: '+name)
   blob=run('hash-object','-w','--stdin',input=data)
   run('update-index','--add','--cacheinfo','100644',blob,name,env=env)
  tree=run('write-tree',env=env)
  if tree!=TREE: raise ValueError('Full repository tree mismatch')
  commit=run('hash-object','-t','commit','-w','--stdin',input=RAW_COMMIT)
  if commit!=COMMIT: raise ValueError('Original commit identity mismatch')
  if run('rev-parse',COMMIT+'^')!=BASE: raise ValueError('Unexpected parent')
  print(json.dumps({'result':'EXACT_TESTED_COMMIT_IMPORTED','commit':commit,'tree':tree,'files':33,'model_calls':0,'notebook_executions':0}),flush=True)
if __name__=='__main__':
 import sys
 main(sys.argv[1])
