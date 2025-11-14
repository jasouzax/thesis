"use strict";
var __assign = (this && this.__assign) || function () {
    __assign = Object.assign || function(t) {
        for (var s, i = 1, n = arguments.length; i < n; i++) {
            s = arguments[i];
            for (var p in s) if (Object.prototype.hasOwnProperty.call(s, p))
                t[p] = s[p];
        }
        return t;
    };
    return __assign.apply(this, arguments);
};
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.Puppet = exports.Automater = void 0;
var electron_1 = require("electron");
var Automater = /** @class */ (function () {
    function Automater(window) {
        this.window = window;
    }
    Automater.prototype.session = function (name) {
        if (name === void 0) { name = null; }
        var view = new electron_1.BrowserView({
            webPreferences: __assign({ contextIsolation: true, nodeIntegration: false, devTools: true, sandbox: true }, (name == null ? {} : { session: electron_1.session.fromPartition("persist:".concat(name)) }))
        });
        // attach primary view
        this.window.setBrowserView(view);
        var _a = this.window.getContentSize(), w = _a[0], h = _a[1];
        view.setBounds({ x: 0, y: 0, width: w, height: h });
        // open devtools when requested
        if (view.webContents.isDevToolsOpened() === false) {
            view.webContents.openDevTools({ mode: 'detach' });
        }
        return new Puppet(view);
    };
    Automater.prototype.host = function (req, res) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
                res.end('Simple HTTP Server\n');
                return [2 /*return*/];
            });
        });
    };
    return Automater;
}());
exports.Automater = Automater;
var Puppet = /** @class */ (function () {
    function Puppet(view) {
        this.view = view;
    }
    /** Failure */
    Puppet.prototype.error = function (e) {
        console.error('Automation Error: ', e);
    };
    /** Goes to a URL */
    Puppet.prototype.go = function (url) {
        return __awaiter(this, void 0, void 0, function () {
            var e_1;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 2, , 3]);
                        this.view.webContents.loadURL(url);
                        return [4 /*yield*/, this.load()];
                    case 1:
                        _a.sent();
                        return [3 /*break*/, 3];
                    case 2:
                        e_1 = _a.sent();
                        this.error(e_1);
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /** Wait for page to load */
    Puppet.prototype.load = function () {
        var _this = this;
        return new Promise(function (res) { return _this.view.webContents.on('did-finish-load', function () { return res(); }); });
    };
    /** Execute Javascript */
    Puppet.prototype.exec = function (code) {
        return __awaiter(this, void 0, void 0, function () {
            var e_2;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        _a.trys.push([0, 2, , 3]);
                        return [4 /*yield*/, this.view.webContents.executeJavaScript(code)];
                    case 1: return [2 /*return*/, _a.sent()];
                    case 2:
                        e_2 = _a.sent();
                        this.error(e_2);
                        return [3 /*break*/, 3];
                    case 3: return [2 /*return*/];
                }
            });
        });
    };
    /** Get title */
    Puppet.prototype.title = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.exec('document.title')];
                    case 1: return [2 /*return*/, _a.sent()];
                }
            });
        });
    };
    /** Wait for query */
    Puppet.prototype.wait = function (query) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.exec("\n            new Promise(resolve => {\n                const checkElement = setInterval(() => {\n                    if (document.querySelector('".concat(query, "')) {\n                        clearInterval(checkElement);\n                        resolve();\n                    }\n                }, 100);\n            });\n        "))];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    /** Type to value */
    Puppet.prototype.type = function (query, text) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.exec("\n            document.querySelector('".concat(query, "').value = '").concat(text, "';\n        "))];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    /** Click query */
    Puppet.prototype.click = function (query) {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0: return [4 /*yield*/, this.exec("\n            document.querySelector('".concat(query, "').click();\n        "))];
                    case 1:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    return Puppet;
}());
exports.Puppet = Puppet;
//# sourceMappingURL=automater.js.map