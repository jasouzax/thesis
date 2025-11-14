import { app, BrowserWindow, session, BrowserView } from 'electron';
import * as http from 'http';

export class Automater {
    window:BrowserWindow;
    constructor(window:BrowserWindow) {
        this.window = window;
    }
    session(name:string|null=null) {
        const view = new BrowserView({
            webPreferences: {
                contextIsolation: true,
                nodeIntegration: false,
                devTools: true,
                sandbox: true,
                ...(name == null ? {} : { session: session.fromPartition(`persist:${name}`) })
            }
        });

        // attach primary view
        this.window.setBrowserView(view);
        const [w, h] = this.window.getContentSize();
        view.setBounds({ x: 0, y: 0, width: w, height: h });

        // open devtools when requested
        if (view.webContents.isDevToolsOpened() === false) {
            view.webContents.openDevTools({ mode: 'detach' });
        }

        return new Puppet(view);
    }
    async host(req:http.IncomingMessage, res:http.ServerResponse<http.IncomingMessage>) {
        res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Simple HTTP Server\n');
    }

}
export class Puppet {
    view:BrowserView;
    constructor(view:BrowserView) {
        this.view = view;
    }
    /** Failure */
    error(e:Error) {
        console.error('Automation Error: ', e);
    }
    /** Goes to a URL */
    async go(url:string) {
        try {
            this.view.webContents.loadURL(url);
            await this.load();
        } catch (e) {
            this.error(e);
        }
    }
    /** Wait for page to load */
    load() {
        return new Promise<void>(res => this.view.webContents.on('did-finish-load', () => res()));
    }
    /** Execute Javascript */
    async exec<T extends any>(code:string):Promise<T> {
        try {
            return await this.view.webContents.executeJavaScript(code) as Promise<T>;
        } catch (e) {
            this.error(e);
        }
    }
    /** Get title */
    async title():Promise<string> {
        return await this.exec<string>('document.title');
    }
    /** Wait for query */
    async wait(query:string) {
        await this.exec(`
            new Promise(resolve => {
                const checkElement = setInterval(() => {
                    if (document.querySelector('${query}')) {
                        clearInterval(checkElement);
                        resolve();
                    }
                }, 100);
            });
        `);
    }
    /** Type to value */
    async type(query:string, text:string) {
        await this.exec(`
            document.querySelector('${query}').value = '${text}';
        `);
    }
    /** Click query */
    async click(query:string) {
        await this.exec(`
            document.querySelector('${query}').click();
        `);
    }
}