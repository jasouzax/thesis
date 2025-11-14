/* ----- IMPORTS ----- */
import { app, BrowserWindow, WebContentsView, session } from 'electron';
import { App } from './app';
import * as http from 'http';

/* ----- VARIABLES ---- */
/** Main Window */
let main:BrowserWindow|null = null;
let mainApp:App|null = null;

/* ----- MAIN CODE ---- */
app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});
app.on('ready', async () => {


    // Main Window Setup
    main = new BrowserWindow({
        width: App.width, height: App.height,
        webPreferences: {
            contextIsolation: true,
            nodeIntegration: false
        }
    });
    main.loadURL('about:blank');
    main.on('closed', () => {
        main = null;
    });
    mainApp = new App(main);


    // Server
    const server = http.createServer(async (req, res) => {
        await mainApp.host(req, res);
    });
    server.listen(App.port, '0.0.0.0');
    app.on('will-quit', () => {
        server.close(() => console.log('HTTP server closed'));
    });



    await mainApp.main();
});