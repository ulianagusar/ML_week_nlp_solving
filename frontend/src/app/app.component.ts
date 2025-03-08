// import { Component, OnInit } from '@angular/core';
// import { HttpClientModule } from '@angular/common/http';
// import { HttpClient } from '@angular/common/http';
// import { CommonModule } from '@angular/common'; 

// @Component({
//   selector: 'app-root',
//   standalone: true,  
//   imports: [CommonModule, HttpClientModule],  
//   templateUrl: './app.component.html',
//   styleUrls: ['./app.component.css']
// })
// export class AppComponent implements OnInit {
//   title = 'my-angular-project';
//   posts: any[] = [];  
//   filteredPosts: any[] = []; 
//   model: any[] = [];  
//   filteredModels: any[] = []; 
//   channels: string[] = ['Вертолатте', 'ДРОННИЦА', 'Донбасс Россия']; 
//   models: string[] = ['ruBert', 'xgboost'];
//   selectedChannel: string = 'all';  
//   selectedModels: string = 'all';
//   selectedStartDate: string | undefined;
//   selectedEndDate: string | undefined;

//   constructor(private http: HttpClient) {}

//   ngOnInit(): void {
//     // Ініціалізація або виконання додаткових операцій, якщо потрібно
//   }

//   onChannelChange(event: any): void {
//     this.selectedChannel = event.target.value;

//     if (this.selectedChannel === 'all') {
//       this.filteredPosts = this.posts; 
//     } else {
//       this.filteredPosts = this.posts.filter(post => post.Channel === this.selectedChannel);
//     }
//   }

//   onModelsChange(event: any): void {
//     this.selectedModels = event.target.value;

//     if (this.selectedModels === 'all') {
//       this.filteredModels = this.model; 
//     } else {
//       this.filteredModels = this.model.filter(post => post.models === this.selectedModels);
//     }
//   }

//   onStartDateChange(event: any): void {
//     this.selectedStartDate = event.target.value;
//   }

//   onEndDateChange(event: any): void {
//     this.selectedEndDate = event.target.value;
//   }

//   onSearch(): void {
//     const requestBody = {
//       channel: this.selectedChannel,
//       start_date: this.selectedStartDate,
//       end_date: this.selectedEndDate,
//       model: this.selectedModels
//     };
//     console.log(requestBody)
//     // Надсилаємо POST запит для отримання повідомлень
//     this.http.post('http://127.0.0.1:5001/api/fetch_posts', requestBody)
//       .subscribe({
//         next: () => {
//           // Після успішного виконання POST запиту виконуємо GET запит для отримання повідомлень
//           this.http.get<any[]>('http://127.0.0.1:5001/api/posts')
//             .subscribe(data => {
//               console.log(data);
//               this.posts = data;
//               this.filteredPosts = data;  // Оновлюємо список фільтрованих повідомлень
//             }, error => {
//               console.error('Помилка при отриманні постів:', error);
//             });
//         },
//         error: (error) => {
//           console.error('Помилка при виконанні POST запиту:', error);
//         }
//       });
//   }
// }


import { Component, OnInit } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, HttpClientModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'my-angular-project';
  posts: any[] = [];
  filteredPosts: any[] = [];
  channels: string[] = ['Вертолатте', 'ДРОННИЦА', 'Донбасс Россия'];
  models: string[] = ['ruBert', 'xgboost'];
  selectedChannel: string = 'all';
  selectedModel: string = 'all';
  selectedStartDate: string | undefined;
  selectedEndDate: string | undefined;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.fetchPosts();
  }

  fetchPosts(): void {
    this.http.get<any[]>('http://127.0.0.1:5001/api/posts')
      .subscribe({
        next: (data) => {
          this.posts = data;
          this.filteredPosts = data;
        },
        error: (error) => {
          console.error('Помилка при отриманні постів:', error);
        }
      });
  }

  onChannelChange(event: any): void {
    this.selectedChannel = event.target.value;
    this.applyFilters();
  }

  onModelChange(event: any): void {
    this.selectedModel = event.target.value;
    this.applyFilters();
  }

  onStartDateChange(event: any): void {
    this.selectedStartDate = event.target.value;
    this.applyFilters();
  }

  onEndDateChange(event: any): void {
    this.selectedEndDate = event.target.value;
    this.applyFilters();
  }

  applyFilters(): void {
    this.filteredPosts = this.posts.filter(post => {
      return (this.selectedChannel === 'all' || post.Channel === this.selectedChannel) &&
             (this.selectedModel === 'all' || post.Model === this.selectedModel) &&
             (!this.selectedStartDate || new Date(post.MessageDate) >= new Date(this.selectedStartDate)) &&
             (!this.selectedEndDate || new Date(post.MessageDate) <= new Date(this.selectedEndDate));
    });
  }

  onSearch(): void {
    const requestBody = {
      channel: this.selectedChannel !== 'all' ? this.selectedChannel : null,
      start_date: this.selectedStartDate || null,
      end_date: this.selectedEndDate || null,
      model: this.selectedModel !== 'all' ? this.selectedModel : null
    };

    console.log('Відправка запиту:', requestBody);
    
    this.http.post('http://127.0.0.1:5001/api/fetch_posts', requestBody)
      .subscribe({
        next: () => this.fetchPosts(),
        error: (error) => console.error('Помилка при виконанні POST запиту:', error)
      });
  }

  onDownloadCSV(): void {
    this.http.get('http://127.0.0.1:5001/api/get_report', { responseType: 'blob' })
      .subscribe({
        next: (blob) => {
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = 'ODCR.csv';
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          window.URL.revokeObjectURL(url);
        },
        error: (error) => {
          console.error('Помилка при завантаженні CSV:', error);
        }
      });
  }
}

// // import { Component, OnInit } from '@angular/core';
// // import { HttpClientModule } from '@angular/common/http';
// // import { HttpClient } from '@angular/common/http';
// // import { CommonModule } from '@angular/common'; 

// // @Component({
// //   selector: 'app-root',
// //   standalone: true,  
// //   imports: [CommonModule, HttpClientModule],  
// //   templateUrl: './app.component.html',
// //   styleUrls: ['./app.component.css']
// // })
// // export class AppComponent implements OnInit {
// //   title = 'my-angular-project';
// //   posts: any[] = [];  
// //   filteredPosts: any[] = []; 
// //   model: any[] = [];  
// //   filteredModels: any[] = []; 
// //   channels: string[] = ['Вертолатте', 'ДРОННИЦА', 'Донбасс Россия']; 
// //   models: string[] = ['ruBert', 'xgboost'];
// //   selectedChannel: string = 'all';  
// //   selectedModels: string = 'all';
// //   selectedStartDate: string | undefined;
// //   selectedEndDate: string | undefined;

// //   constructor(private http: HttpClient) {}

// //   ngOnInit(): void {
// //     // Ініціалізація або виконання додаткових операцій, якщо потрібно
// //   }

// //   onChannelChange(event: any): void {
// //     this.selectedChannel = event.target.value;

// //     if (this.selectedChannel === 'all') {
// //       this.filteredPosts = this.posts; 
// //     } else {
// //       this.filteredPosts = this.posts.filter(post => post.Channel === this.selectedChannel);
// //     }
// //   }

// //   onModelsChange(event: any): void {
// //     this.selectedModels = event.target.value;

// //     if (this.selectedModels === 'all') {
// //       this.filteredModels = this.model; 
// //     } else {
// //       this.filteredModels = this.model.filter(post => post.models === this.selectedModels);
// //     }
// //   }

// //   onStartDateChange(event: any): void {
// //     this.selectedStartDate = event.target.value;
// //   }

// //   onEndDateChange(event: any): void {
// //     this.selectedEndDate = event.target.value;
// //   }

// //   onSearch(): void {
// //     const requestBody = {
// //       channel: this.selectedChannel,
// //       start_date: this.selectedStartDate,
// //       end_date: this.selectedEndDate,
// //       model: this.selectedModels
// //     };
// //     console.log(requestBody)
// //     // Надсилаємо POST запит для отримання повідомлень
// //     this.http.post('http://127.0.0.1:5001/api/fetch_posts', requestBody)
// //       .subscribe({
// //         next: () => {
// //           // Після успішного виконання POST запиту виконуємо GET запит для отримання повідомлень
// //           this.http.get<any[]>('http://127.0.0.1:5001/api/posts')
// //             .subscribe(data => {
// //               console.log(data);
// //               this.posts = data;
// //               this.filteredPosts = data;  // Оновлюємо список фільтрованих повідомлень
// //             }, error => {
// //               console.error('Помилка при отриманні постів:', error);
// //             });
// //         },
// //         error: (error) => {
// //           console.error('Помилка при виконанні POST запиту:', error);
// //         }
// //       });
// //   }
// // }


// import { Component, OnInit } from '@angular/core';
// import { HttpClientModule } from '@angular/common/http';
// import { HttpClient } from '@angular/common/http';
// import { CommonModule } from '@angular/common';

// @Component({
//   selector: 'app-root',
//   standalone: true,
//   imports: [CommonModule, HttpClientModule],
//   templateUrl: './app.component.html',
//   styleUrls: ['./app.component.css']
// })
// export class AppComponent implements OnInit {
//   title = 'my-angular-project';
//   posts: any[] = [];
//   filteredPosts: any[] = [];
//   channels: string[] = ['Вертолатте', 'ДРОННИЦА', 'Донбасс Россия'];
//   models: string[] = ['ruBert', 'xgboost'];
//   selectedChannel: string = 'all';
//   selectedModel: string = 'all';
//   selectedStartDate: string | undefined;
//   selectedEndDate: string | undefined;

//   constructor(private http: HttpClient) {}

//   ngOnInit(): void {
//     this.fetchPosts();
//   }

//   fetchPosts(): void {
//     console.log("get")
//     this.http.get<any[]>('http://backend1:5001/api/posts')
//       .subscribe({
//         next: (data) => {
//           this.posts = data;
//           this.filteredPosts = data;
//         },
//         error: (error) => {
//           console.error('Помилка при отриманні постів:', error);
//         }
//       });
//   }

//   onChannelChange(event: any): void {
//     this.selectedChannel = event.target.value;
//     this.applyFilters();
//   }

//   onModelChange(event: any): void {
//     this.selectedModel = event.target.value;
//     this.applyFilters();
//   }

//   onStartDateChange(event: any): void {
//     this.selectedStartDate = event.target.value;
//     this.applyFilters();
//   }

//   onEndDateChange(event: any): void {
//     this.selectedEndDate = event.target.value;
//     this.applyFilters();
//   }

//   applyFilters(): void {
//     this.filteredPosts = this.posts.filter(post => {
//       return (this.selectedChannel === 'all' || post.Channel === this.selectedChannel) &&
//              (this.selectedModel === 'all' || post.Model === this.selectedModel) &&
//              (!this.selectedStartDate || new Date(post.MessageDate) >= new Date(this.selectedStartDate)) &&
//              (!this.selectedEndDate || new Date(post.MessageDate) <= new Date(this.selectedEndDate));
//     });
//   }

//   onSearch(): void {
//     const requestBody = {
//       channel: this.selectedChannel !== 'all' ? this.selectedChannel : null,
//       start_date: this.selectedStartDate || null,
//       end_date: this.selectedEndDate || null,
//       model: this.selectedModel !== 'all' ? this.selectedModel : null
//     };

//     console.log('Відправка запиту:', requestBody);
    
//     this.http.post('http://backend1:5001/api/fetch_posts', requestBody)
//       .subscribe({
//         next: () => this.fetchPosts(),
//         error: (error) => console.error('Помилка при виконанні POST запиту:', error)
//       });
//   }

//   onDownloadCSV(): void {
//     this.http.get('http://backend1:5001/api/get_report', { responseType: 'blob' })
//       .subscribe({
//         next: (blob) => {
//           const url = window.URL.createObjectURL(blob);
//           const a = document.createElement('a');
//           a.href = url;
//           a.download = 'ODCR.csv';
//           document.body.appendChild(a);
//           a.click();
//           document.body.removeChild(a);
//           window.URL.revokeObjectURL(url);
//         },
//         error: (error) => {
//           console.error('Помилка при завантаженні CSV:', error);
//         }
//       });
//   }
// }
