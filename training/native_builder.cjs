// Keep the official SDK initialized while candidates are built. Initialization
// performs simulator cleanup, so it must happen once, before training starts.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const nativeDir = process.env.BOOSTER_NATIVE_DIR || (process.platform === 'darwin'
  ? '/Applications/Booster Studio.app/Contents/Resources/app/booster-native'
  : 'C:/Users/lifat/BoosterStudio/resources/app/booster-native');
const dataDir = process.env.BOOSTER_DATA_DIR || (process.platform === 'darwin'
  ? path.join(os.homedir(), 'Library/Application Support/Booster Studio')
  : path.join(process.env.APPDATA || 'C:/Users/lifat/AppData/Roaming', 'Booster Studio'));
const robot = process.env.BOOSTER_ROBOT || '7gofn';
const projectId = crypto.createHash('md5').update(root).digest('hex');
if (process.argv.includes('--check')) {
  if (!fs.existsSync(path.join(nativeDir, 'index.cjs'))) throw Error(`SDK missing: ${nativeDir}`);
  if (!fs.existsSync(path.join(dataDir, 'database/data.db'))) throw Error(`Studio data missing: ${dataDir}`);
  console.log(JSON.stringify({nativeDir, dataDir, robot, projectId, root}));
  process.exit(0);
}
const native = require(path.join(nativeDir, 'index.cjs'));
const ipcArg = process.argv.indexOf('--ipc');
const ipc = ipcArg >= 0 ? path.resolve(process.argv[ipcArg + 1]) : path.join(root, 'build', 'match-training', 'native');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function checked(raw) {
  const response = JSON.parse(raw);
  if (response.errorCode) throw Error(response.error);
  return response.result;
}
async function main() {
  fs.mkdirSync(ipc, {recursive: true});
  await native.init(path.join(nativeDir, 'statics'), dataDir, path.join(ipc, 'logs'), true);
  checked(await native.agentDevActivate(JSON.stringify({projectId, hostSourcePath:root, robotModel:'Booster K1'})));
  checked(await native.activateRobot(robot, 'football_match_pitch_6_K1'));
  fs.writeFileSync(path.join(ipc, 'ready.json'), JSON.stringify({pid:process.pid, time:Date.now()}));
  while (!fs.existsSync(path.join(ipc, 'stop'))) {
    const request = path.join(ipc, 'request.json');
    if (!fs.existsSync(request)) { await pause(500); continue; }
    const job = JSON.parse(fs.readFileSync(request, 'utf8'));
    fs.unlinkSync(request);
    let result;
    try {
      result = {id:job.id, success:true, result:checked(await native.agentDevBuild(JSON.stringify({projectId, robotModel:'Booster K1'})))};
      if (job.deploy) checked(await native.agentDevDeploy(JSON.stringify({projectId,
        robotSeriesNumber:robot, agentId:'com.example.lqinternalmatch'})));
    } catch (error) { result = {id:job.id, success:false, error:String(error)}; }
    fs.writeFileSync(path.join(ipc, `${job.id}.json`), JSON.stringify(result));
  }
}
main().then(() => process.exit(0)).catch(error => {console.error(error); process.exit(1);});
