const API = {
  base: "http://localhost:8000/api/v1",

  authHeaders(){
    try{
      const t = JSON.parse(localStorage.getItem('auth'));
      if(!t || !t.access_token) return {};
      return { 'Authorization': `Bearer ${t.access_token}` };
    }catch{ return {}; }
  },

  async register(email, password, full_name, role="student"){
    const res = await fetch(`${this.base}/auth/register`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, password, full_name, role})});
    if(!res.ok) throw new Error("register failed");
    return res.json();
  },
  async login(email, password){
    const res = await fetch(`${this.base}/auth/login`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, password})});
    if(!res.ok) throw new Error("login failed");
    return res.json();
  },
  async vectorSearch(text, k=5){
    const res = await fetch(`${this.base}/vectors/search`, {method:"POST", headers:{'Content-Type':'application/json', ...this.authHeaders()}, body: JSON.stringify({text, k})});
    return res.json();
  },
  async teacherRequests(){
    const res = await fetch(`${this.base}/teacher/students/requests`, { headers: { ...this.authHeaders() }});
    return res.json();
  },
  async approveRequest(request_id, approve){
    const res = await fetch(`${this.base}/teacher/students/approve?request_id=${request_id}&approve=${approve}`, {method:"POST", headers:{ ...this.authHeaders() }});
    return res.json();
  },
  async listAssignmentsTeacher(){
    const res = await fetch(`${this.base}/teacher/assignments`, { headers: { ...this.authHeaders() }});
    return res.json();
  },
  async createAssignment(payload){
    const res = await fetch(`${this.base}/teacher/assignments`, {method:"POST", headers:{'Content-Type':'application/json', ...this.authHeaders()}, body: JSON.stringify(payload)});
    return res.json();
  },
  async listAssignmentsStudent(){
    const res = await fetch(`${this.base}/student/assignments`, { headers: { ...this.authHeaders() }});
    return res.json();
  },
  async sendRequestTeacher(teacher_id){
    const res = await fetch(`${this.base}/student/request-teacher?teacher_id=${teacher_id}`, {method:"POST", headers:{ ...this.authHeaders() }});
    return res.json();
  }
};
