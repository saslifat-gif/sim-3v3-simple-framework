// Keep the official SDK initialized while candidates are built. Initialization
// performs simulator cleanup, so it must happen once, before training starts.
const fs = require('node:fs');
const path = require('node:path');
const native = require('C:/Users/lifat/BoosterStudio/resources/app/booster-native/index.cjs');
const root = path.resolve(__dirname, '..');
const ipcArg = process.argv.indexOf('--ipc');
const ipc = ipcArg >= 0 ? path.resolve(process.argv[ipcArg + 1]) : path.join(root, 'build', 'match-training', 'native');
const projectId = '1eb4ef2862284a0e2bb54ab934df3dcc';
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
function checked(raw) {
  const response = JSON.parse(raw);
  if (response.errorCode) throw Error(response.error);
  return response.result;
}
async function main() {
  fs.mkdirSync(ipc, {recursive: true});
  await native.init('C:/Users/lifat/BoosterStudio/resources/app/booster-native/statics',
    'C:/Users/lifat/AppData/Roaming/Booster Studio', path.join(ipc, 'logs'), true);
  checked(await native.agentDevActivate(JSON.stringify({projectId, hostSourcePath:root, robotModel:'Booster K1'})));
  checked(await native.activateRobot('7gofn', 'football_match_pitch_6_K1'));
  fs.writeFileSync(path.join(ipc, 'ready.json'), JSON.stringify({pid:process.pid, time:Date.now()}));
  while (!fs.existsSync(path.join(ipc, 'stop'))) {
    const request = path.join(ipc, 'request.json');
    if (!fs.existsSync(request)) { await pause(500); continue; }
    const job = JSON.parse(fs.readFileSync(request, 'utf8'));
    fs.unlinkSync(request);
    let result;
    try {
      result = {id:job.id, success:true, result:checked(await native.agentDevBuild(JSON.stringify({projectId, robotModel:'Booster K1'})))};
    } catch (error) { result = {id:job.id, success:false, error:String(error)}; }
    fs.writeFileSync(path.join(ipc, `${job.id}.json`), JSON.stringify(result));
  }
}
main().then(() => process.exit(0)).catch(error => {console.error(error); process.exit(1);});
