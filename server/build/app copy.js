"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var electron_1 = require("electron");
var mainWindow = null;
electron_1.app.on('window-all-closed', function () {
    if (process.platform !== 'darwin')
        electron_1.app.quit();
});
electron_1.app.on('ready', function () {
    mainWindow = new electron_1.BrowserWindow({
        width: 1024,
        height: 768,
        webPreferences: {
            // @ts-ignore
            webviewTag: true,
            contextIsolation: true,
            nodeIntegration: false,
        },
    });
    mainWindow.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(/*html*/ "\n        <!doctype html>\n        <html>\n            <head>\n                <meta charset=\"utf-8\">\n                <style>\n                    html,body{height:100%;margin:0}\n                    webview{width:500px;height:500px;display:block;border:1px solid red}\n                </style>\n            </head>\n            <body>\n                <webview id=\"web\" src=\"https://facebook.com\" allowpopups></webview>\n            </body>\n        </html>"));
    mainWindow.on('closed', function () {
        mainWindow = null;
    });
});
//# sourceMappingURL=app%20copy.js.map