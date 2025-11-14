export interface Credential {
    [web:string]:{
        'user':string,
        'pass':string,
        'pin'?:string
    }
}
export class User {

}