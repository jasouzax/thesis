import { Automater, Puppet } from './automater';
import { IncomingMessage, ServerResponse } from 'http';

export class App extends Automater {
    /* ----- CONFIGURATIONS ----- */
    /** Default window width */
    static width = 1024;
    /** Default window height */
    static height = 768;
    /** Port of server */
    static port = 80;
    /* ----- MAIN CODE ----- */
    john:Puppet|null = null;
    /** Main code */
    async main() {
        // Session
        const john = this.session();
        await john.go('https://facebook.com');
        const title = await john.title();
        if (title == 'Facebook – log in or sign up') {
            await john.wait('input[id="email"]');
            await john.type('input[id="email"]', 'jasondsouza2003@gmail.com');
            await john.type('input[id="pass"]', '$cimaticsSocial23FB');
            await john.click('[type="submit"]')
            await john.load();
        }
        await john.go('https://facebook.com/messages');
        this.john = john;
    }
    /** Server */
    async host(req:IncomingMessage, res:ServerResponse<IncomingMessage>) {
        const parsed = new URL(req.url || '/', `http://localhost:${App.port}`);
        if (parsed.pathname == '/api') {
            if (this.john == null) {
                res.writeHead(505, { 'Content-Type': 'text/plain; charset=utf-8' });
                res.end('500 Internal Server Error')
            } else {
                const query = Object.fromEntries(parsed.searchParams.entries());
                await this.john.go(`https://facebook.com/messages/${query['id']}`);
                if ('call' in query) {
                    await this.john.wait('[aria-label="Start a voice call"]');
                    await this.john.click('[aria-label="Start a voice call"]');
                }
                res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
                res.end('done');
            }
            return;

        }
        res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
        res.end('Simple HTTP Server\n');
    }
}